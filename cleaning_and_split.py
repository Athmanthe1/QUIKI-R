"""
cleaning_and_split.py
Production-grade backend data-processing utility for the Wikimedia Commons
Category Suggestion pipeline.
"""
import json
import logging
import os
import re
import gc
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd

# ==============================================================================
# 1. OPTIONAL HEAVY DEPENDENCIES
# ==============================================================================
try:
    from PIL import Image
    _HASH_AVAILABLE = True          # our own pHash only needs PIL + scipy (no imagehash dependency)
except ImportError:
    _HASH_AVAILABLE = False

# ==============================================================================
# 2. CONFIGURATION & LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("cleaning_and_split")

# Process / maintenance category detection.
#
# PREFERRED: the dump's hidden-category flag (page_props.hiddencat). When the caller
# supplies `hidden_categories`, exactly those categories are dropped and the regex below
# is NOT consulted.
#
# FALLBACK (no flag available): a deliberately NARROW set of anchored patterns for
# well-known Commons process categories. They are anchored to the start of the normalised
# name so that real content categories such as "Aircraft maintenance" or "Beach cleanup"
# are kept. Names are normalised first: "Category:" removed, underscores -> spaces,
# whitespace collapsed, case-folded.
_PROCESS_FALLBACK_PATTERNS: List[re.Pattern] = [re.compile(p, re.I) for p in (
    r"^(files? )?uploaded (with|via|by|from) ",                       # Uploaded with UploadWizard / via Campaign:...
    r"^media (needing|without|lacking|missing|with) ",                # Media needing categories as of ...
    r"^files? with (no|missing|broken|invalid) ",                     # Files with no machine-readable author
    r"^files? (needing|lacking|missing) ",
    r"^(cc[- ]by|cc[- ]zero|cc0|pd[- ]|gfdl|self[- ]published work|attribution)",   # licence-template categories
    r"^licen[sc]e (review|template|tag)",
    r"^(possible )?copyright (violations?|status)",
    r"^(speedy|requests for) deletion",
    r"^(commons|wikimedia commons) (maintenance|cleanup|backlog)",
    r"^(maintenance|cleanup|clean-up|backlog)( categories| of| for|$| \d)",
    r"^(maintenance|cleanup|clean-up|backlog)[ -](maintenance|cleanup|clean-up|backlog|tracking|categories)",
    r"^(hidden|tracking|process|maintenance) categories$",
    r"^(empty|deleted) (categories|pages|files|galleries)$",
    r" as of \d{1,2} \w+ \d{4}$",                                    # dated backlog buckets
    r" as of \d{4}$",
    r"^unreviewed ",
    r"^(files|media|images) (to be|pending) (checked|reviewed|categori[sz]ed)",
)]
# Back-compat alias: older code/tests import this name.
MAINTENANCE_PATTERNS = _PROCESS_FALLBACK_PATTERNS

CONTENT_SAFELIST_PREFIXES: Tuple[str, ...] = ("Commons:", "File:", "Image:")
DEFAULT_HAMMING_THRESHOLD: int = 8
DEFAULT_SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)
COSINE_SIM_THRESHOLD: float = 0.95

# ==============================================================================
# 3. DIAGNOSTICS REPORT
# ==============================================================================
@dataclass
class CleaningReport:
    total_records: int = 0
    records_with_categories: int = 0
    records_without_categories: int = 0
    maintenance_categories_removed: int = 0
    duplicate_hashes_removed: int = 0
    duplicate_phash_removed: int = 0
    duplicate_embedding_removed: int = 0
    uploader_overlap_resolved: int = 0
    temporal_dropped: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 50, "  CLEANING REPORT", "=" * 50,
            f"  Total records ingested        : {self.total_records}",
            f"  Records with >=1 category     : {self.records_with_categories}",
            f"  Records without any category  : {self.records_without_categories}",
            f"  Maintenance cats stripped     : {self.maintenance_categories_removed}",
            f"  Near-duplicates removed (all) : {self.duplicate_hashes_removed}",
            f"    by perceptual hash          : {self.duplicate_phash_removed}",
            f"    by embedding similarity     : {self.duplicate_embedding_removed}",
            f"  Records reassigned (grouping) : {self.uploader_overlap_resolved}",
            f"  Dropped (outside split window): {self.temporal_dropped}",
            "=" * 50,
        ]
        return "\n".join(lines)

# ==============================================================================
# 4. LOW-LEVEL HELPERS & CLEANING
# ==============================================================================
def _safe_str(value: Any) -> str:
    if value is None or isinstance(value, (list, tuple, dict, set, np.ndarray)):
        return ""
    try:
        if pd.isna(value): return ""
    except (TypeError, ValueError): pass
    return str(value).strip()

def normalize_category_name(category: str) -> str:
    """Comparison form: no 'Category:' prefix, underscores -> spaces, collapsed whitespace, case-folded."""
    bare = _safe_str(category)
    for prefix in ("Category:", "category:", "CATEGORY:"):
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
            break
    return re.sub(r"\s+", " ", bare.replace("_", " ")).strip().casefold()


def canonical_category_name(category: str) -> str:
    """Display/ID form used for class lists: no prefix, spaces not underscores, original case kept."""
    bare = _safe_str(category)
    for prefix in ("Category:", "category:", "CATEGORY:"):
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
            break
    return re.sub(r"\s+", " ", bare.replace("_", " ")).strip()


def normalize_hidden_set(hidden_categories: Optional[Iterable[str]]) -> Optional[Set[str]]:
    """Normalise a hidden-category collection once. None means 'no flag information available'."""
    if hidden_categories is None:
        return None
    return {normalize_category_name(c) for c in hidden_categories}


def is_process_category(category: str, hidden_categories: Optional[Set[str]] = None) -> bool:
    """True when `category` encodes process, not content.

    `hidden_categories` must already be normalised (see normalize_hidden_set). If it is not
    None the dump's flag is authoritative and the regex fallback is skipped.
    """
    if not category:
        return False
    norm = normalize_category_name(category)
    if hidden_categories is not None:
        return norm in hidden_categories
    raw = _safe_str(category)
    bare = raw[len("Category:"):] if raw.lower().startswith("category:") else raw
    if any(bare.strip().startswith(safe) for safe in CONTENT_SAFELIST_PREFIXES):
        return False
    return any(pat.search(norm) for pat in _PROCESS_FALLBACK_PATTERNS)


def _is_maintenance_category(category: str) -> bool:
    """Back-compat wrapper (regex fallback only)."""
    return is_process_category(category, None)


def _clean_category_list(categories: Sequence[str], report: CleaningReport,
                         hidden_categories: Optional[Iterable[str]] = None,
                         canonicalize: bool = False,
                         _hidden_normalized: bool = False) -> List[str]:
    if not categories: return []
    hidden = hidden_categories
    if hidden is not None and not _hidden_normalized:
        hidden = normalize_hidden_set(hidden)
    seen, cleaned = set(), []
    for raw in categories:
        cat = _safe_str(raw)
        if not cat: continue
        if is_process_category(cat, hidden):
            report.maintenance_categories_removed += 1
            continue
        out = canonical_category_name(cat) if canonicalize else cat
        key = normalize_category_name(out) if canonicalize else out
        if key not in seen:
            seen.add(key)
            cleaned.append(out)
    return cleaned

def clean_records(raw_df: pd.DataFrame, report: CleaningReport,
                  category_col: str = "categories", timestamp_col: str = "timestamp",
                  uploader_col: str = "uploader",
                  hidden_categories: Optional[Iterable[str]] = None,
                  canonicalize: bool = False) -> Tuple[pd.DataFrame, CleaningReport]:
    if raw_df is None or len(raw_df) == 0:
        return pd.DataFrame(), report
        
    df = raw_df.copy()
    report.total_records += len(df)
    hidden_norm = normalize_hidden_set(hidden_categories)   # normalise once, not per row
    
    df = df.dropna(subset=[timestamp_col, uploader_col])
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    df = df.dropna(subset=[timestamp_col])
    
    df[uploader_col] = df[uploader_col].map(_safe_str).astype('category')
    df = df[df[uploader_col] != ""]
    
    if category_col not in df.columns:
        df[category_col] = [[] for _ in range(len(df))]
        
    def _clean_row(cats: Any) -> List[str]:
        if cats is None or isinstance(cats, (int, float, bool)):
            return []
        if isinstance(cats, str):
            s = cats.strip()
            if not s: return []
            if s.startswith("["):
                try: cats = json.loads(s)
                except (json.JSONDecodeError, TypeError): cats = [s]
            else: cats = [s]
        if not isinstance(cats, (list, tuple, set)):
            return []
        return _clean_category_list(list(cats), report, hidden_norm, canonicalize, _hidden_normalized=True)
        
    df[category_col] = df[category_col].map(_clean_row)
    lengths = df[category_col].map(len)
    
    report.records_with_categories += int((lengths > 0).sum())
    report.records_without_categories += int((lengths == 0).sum())
    
    return df.reset_index(drop=True), report

# ==============================================================================
# 5. INGESTION
# ==============================================================================
def load_raw_records(source: Union[str, Path, pd.DataFrame], chunk_size: int = 10_000) -> Iterator[pd.DataFrame]:
    if isinstance(source, pd.DataFrame):
        yield source
        return
        
    path = Path(source)
    if not path.exists(): raise FileNotFoundError(f"Source not found: {path}")
    
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        for chunk in pd.read_json(path, lines=True, chunksize=chunk_size):
            yield chunk
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        yield pd.DataFrame(data if isinstance(data, list) else [data])
    else:
        raise ValueError(f"Unsupported extension: {suffix}")

# ==============================================================================
# 6. DEDUPLICATION (dual-mode: pHash BEFORE extraction, embeddings AFTER)
# ==============================================================================
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorised popcount of a uint64 array (np.bitwise_count on numpy>=2, table fallback otherwise)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.uint8)
    b = np.ascontiguousarray(x).view(np.uint8).reshape(*x.shape, 8)
    return _POPCOUNT8[b].sum(axis=-1).astype(np.uint8)


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def compute_phash(image_path: str, hash_size: int = 8, highfreq_factor: int = 4) -> Optional[str]:
    """64-bit DCT perceptual hash as a 16-char hex string, or None if the file cannot be read.

    Same recipe as the common 'phash': grayscale -> resize to 32x32 -> 2-D DCT -> keep the
    8x8 low-frequency block -> threshold at its median. Implemented here (numpy + scipy) so the
    pipeline has no extra dependency and the same code runs in tests and in production.
    """
    if not _HASH_AVAILABLE or not image_path or not os.path.isfile(image_path):
        return None
    try:
        from scipy.fft import dctn
        size = hash_size * highfreq_factor
        with Image.open(image_path) as img:
            g = img.convert("L").resize((size, size), Image.LANCZOS)
        d = dctn(np.asarray(g, dtype=np.float64), type=2, norm="ortho")[:hash_size, :hash_size]
        bits = (d > np.median(d)).flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return format(value, "0{}x".format(hash_size * hash_size // 4))
    except Exception as exc:
        logger.warning("pHash failed for %s: %s", image_path, exc)
        return None


def _hex_to_u64(values: Sequence[Optional[str]]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse hex hashes into (uint64 array, valid mask). Missing/invalid hashes are invalid, not exact-matched."""
    arr = np.zeros(len(values), dtype=np.uint64)
    valid = np.zeros(len(values), dtype=bool)
    for i, v in enumerate(values):
        if isinstance(v, str) and v:
            try:
                arr[i] = np.uint64(int(v, 16)); valid[i] = True
            except ValueError:
                pass
    return arr, valid


def hamming_duplicate_flags(hashes: np.ndarray, threshold: int = DEFAULT_HAMMING_THRESHOLD,
                            row_block: int = 512) -> np.ndarray:
    """Flag every hash that is within `threshold` bits of an EARLIER hash (transitively, via union-find).

    Candidate generation uses the pigeonhole principle: split the 64 bits into threshold+1
    chunks; two hashes within `threshold` bits must be identical in at least one chunk. Only
    hashes that share a chunk value are compared, with vectorised XOR+popcount, so the cost is
    far below all-pairs. Exact for any n; no subsampling and no exact-match fallback.
    """
    n = len(hashes)
    hashes = hashes.astype(np.uint64, copy=False)
    parent = {i: i for i in range(n)}                     # dict-based union-find (as before)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:                     # smaller index stays root = "first seen" survives
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb: parent[rb] = ra
            else: parent[ra] = rb

    n_chunks = max(1, threshold + 1)
    bounds = np.linspace(0, 64, n_chunks + 1).astype(int)
    for c in range(n_chunks):
        lo, hi = int(bounds[c]), int(bounds[c + 1])
        mask = np.uint64((1 << (hi - lo)) - 1)
        vals = (hashes >> np.uint64(lo)) & mask
        order = np.argsort(vals, kind="stable")
        sv = vals[order]
        cuts = np.flatnonzero(np.diff(sv)) + 1
        for grp in np.split(order, cuts):
            if len(grp) < 2:
                continue
            gh = hashes[grp]
            for s0 in range(0, len(grp), row_block):
                blk = gh[s0:s0 + row_block]
                dist = _popcount64(blk[:, None] ^ gh[None, :])
                ii, jj = np.nonzero(dist <= threshold)
                for i, j in zip(ii.tolist(), jj.tolist()):
                    a, b = int(grp[s0 + i]), int(grp[j])
                    if a != b:
                        union(a, b)
    return np.array([find(i) != i for i in range(n)], dtype=bool)


def mark_near_duplicates_hashes(df: pd.DataFrame, hash_col: str = "phash",
                                threshold: int = DEFAULT_HAMMING_THRESHOLD) -> pd.DataFrame:
    """Add `is_duplicate` from a column of hex hashes. Rows without a hash are never flagged."""
    arr, valid = _hex_to_u64(df[hash_col].tolist())
    flags = np.zeros(len(df), dtype=bool)
    if valid.any():
        idx = np.flatnonzero(valid)
        flags[idx] = hamming_duplicate_flags(arr[idx], threshold)
    out = df.copy()
    out["is_duplicate"] = flags
    return out


def mark_near_duplicates_phash(df: pd.DataFrame, threshold: int = DEFAULT_HAMMING_THRESHOLD) -> pd.DataFrame:
    """Compute pHash from `image_path`, then flag near-duplicates (kept for API compatibility)."""
    out = df.copy()
    out["phash"] = out["image_path"].map(compute_phash)
    return mark_near_duplicates_hashes(out, "phash", threshold)


def mark_near_duplicates_embeddings(X: np.ndarray, threshold: float = COSINE_SIM_THRESHOLD,
                                    block_size: int = 1024, acc_chunk: int = 50_000) -> np.ndarray:
    """Greedy cosine de-duplication with blocked matrix products. Returns a bool array, True = duplicate.

    A row is kept iff its max cosine similarity to every earlier KEPT row is < threshold
    (identical semantics to the old per-row loop, but one matmul per block). Rows with
    non-finite values are flagged as duplicates (dropped). Memory is O(block_size * acc_chunk).
    """
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    dup = np.zeros(n, dtype=bool)
    if n == 0:
        return dup
    finite = np.isfinite(X).all(axis=1)
    dup[~finite] = True
    X = np.where(finite[:, None], X, 0.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(norms == 0, 1.0, norms)
    acc = np.empty_like(Xn)
    n_acc = 0
    for s0 in range(0, n, block_size):
        e0 = min(s0 + block_size, n)
        blk = Xn[s0:e0]
        cand = finite[s0:e0].copy()
        if n_acc:
            best = np.full(len(blk), -np.inf, dtype=np.float32)
            for a0 in range(0, n_acc, acc_chunk):
                best = np.maximum(best, (blk @ acc[a0:min(a0 + acc_chunk, n_acc)].T).max(axis=1))
            cand &= best < threshold
        G = blk @ blk.T
        kept_local: List[int] = []
        for i in np.flatnonzero(cand).tolist():
            if kept_local and float(G[i, kept_local].max()) >= threshold:
                continue
            kept_local.append(i)
        keep = np.zeros(len(blk), dtype=bool)
        keep[kept_local] = True
        dup[s0:e0][~keep & finite[s0:e0]] = True
        if kept_local:
            acc[n_acc:n_acc + len(kept_local)] = blk[kept_local]
            n_acc += len(kept_local)
    return dup


def remove_near_duplicates_embeddings(df: pd.DataFrame, emb_col: str = "embeddings") -> pd.DataFrame:
    """DataFrame wrapper (kept for API compatibility): adds `is_duplicate` from a list-valued column."""
    out = df.copy()
    if emb_col not in out.columns or out[emb_col].isna().all():
        out["is_duplicate"] = False
        return out
    ok = out[emb_col].map(lambda v: isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0)
    flags = np.ones(len(out), dtype=bool)                      # malformed -> dropped, as before
    if ok.any():
        mat = np.vstack([np.asarray(v, dtype=np.float32) for v in out.loc[ok, emb_col]])
        flags[ok.to_numpy()] = mark_near_duplicates_embeddings(mat)
    out["is_duplicate"] = flags
    return out


def dedup_dataframe(df: pd.DataFrame, hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD,
                    cosine_threshold: float = COSINE_SIM_THRESHOLD,
                    timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Dual-mode dedup (both modes run whenever their input exists).

    1. pHash (needs `image_path` or a `phash` column) - cheap, runs BEFORE embedding extraction.
    2. Embedding cosine (needs an `embeddings` column) - runs on the pHash survivors, AFTER extraction.
    Within each mode the earliest upload (by timestamp, else row order) is the copy that survives.
    Adds dup_phash, dup_embedding and is_duplicate (their OR); the index is preserved.
    """
    out = df.copy()
    out["dup_phash"] = False
    out["dup_embedding"] = False
    order = (np.argsort(pd.to_datetime(out[timestamp_col], utc=True).to_numpy(), kind="stable")
             if timestamp_col in out.columns else np.arange(len(out)))

    if "phash" in out.columns or "image_path" in out.columns:
        if "phash" not in out.columns:
            out["phash"] = out["image_path"].map(compute_phash)
        sub = out.iloc[order]
        flagged = mark_near_duplicates_hashes(sub[["phash"]], "phash", hamming_threshold)["is_duplicate"].to_numpy()
        out.iloc[order, out.columns.get_loc("dup_phash")] = flagged

    if "embeddings" in out.columns and not out["embeddings"].isna().all():
        survivors = [i for i in order if not out["dup_phash"].iat[i]]
        if survivors:
            sub = out.iloc[survivors]
            flagged = remove_near_duplicates_embeddings(sub[["embeddings"]].assign())["is_duplicate"].to_numpy()
            # remove_near_duplicates_embeddings keeps row order of `sub`, which is already time-ordered
            out.iloc[survivors, out.columns.get_loc("dup_embedding")] = flagged

    out["is_duplicate"] = out["dup_phash"] | out["dup_embedding"]
    return out


# ==============================================================================
# 7. LEAKAGE-FREE TEMPORAL & UPLOADER-GROUPED SPLIT
# ==============================================================================
def temporal_group_split(df: pd.DataFrame, ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
                         timestamp_col: str = "timestamp", uploader_col: str = "uploader",
                         report: Optional[CleaningReport] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal AND uploader-disjoint split.

    1. Two time cut-offs T1 <= T2 are taken from the timestamp distribution (by row count).
    2. Each uploader is assigned to exactly ONE split, by the median timestamp of their uploads.
    3. A row is kept only if its timestamp falls inside its split's window
       (train: ts <= T1, val: T1 < ts <= T2, test: ts > T2). Rows of an uploader that fall
       outside the window of the uploader's split are DROPPED (counted in
       report.temporal_dropped), never moved: moving them would break uploader-disjointness.
    4. Final guard: val/test rows not strictly newer than the latest train/val row are dropped,
       so no test image is older than the latest train image.
    """
    df = df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    n = len(df)
    if n == 0:
        empty = df.copy()
        return empty, empty.copy(), empty.copy()
    ts = df[timestamp_col]
    train_end = min(max(int(n * ratios[0]), 0), n)
    val_end = min(max(int(n * (ratios[0] + ratios[1])), train_end), n)
    t1 = ts.iloc[train_end - 1] if train_end > 0 else ts.iloc[0] - pd.Timedelta(days=1)
    t2 = ts.iloc[val_end - 1] if val_end > 0 else t1

    # uploader -> split by median timestamp (convert to int64 ns for a robust median)
    ts_ns = ts.astype("int64")
    med = ts_ns.groupby(df[uploader_col], observed=True).median()
    # cut-offs taken from the SAME int64 representation (its unit depends on the pandas
    # version: ns or us), never from Timestamp.value, which is always ns.
    t1_ns = int(ts_ns.iloc[train_end - 1]) if train_end > 0 else int(ts_ns.iloc[0]) - 1
    t2_ns = int(ts_ns.iloc[val_end - 1]) if val_end > 0 else t1_ns
    split_of = pd.Series(np.where(med <= t1_ns, "train", np.where(med <= t2_ns, "val", "test")), index=med.index)
    assigned = df[uploader_col].map(split_of).astype(object)

    in_window = ((assigned == "train") & (ts <= t1)) | \
                ((assigned == "val") & (ts > t1) & (ts <= t2)) | \
                ((assigned == "test") & (ts > t2))
    df = df.assign(_split=assigned)[in_window].reset_index(drop=True)

    train_df = df[df["_split"] == "train"]
    val_df = df[df["_split"] == "val"]
    test_df = df[df["_split"] == "test"]
    # final guard (also protects against any future change to the window logic)
    if len(train_df):
        latest_train = train_df[timestamp_col].max()
        val_df = val_df[val_df[timestamp_col] > latest_train]
        test_df = test_df[test_df[timestamp_col] > latest_train]
    if len(val_df):
        test_df = test_df[test_df[timestamp_col] > val_df[timestamp_col].max()]

    kept = len(train_df) + len(val_df) + len(test_df)
    if report is not None:
        report.temporal_dropped = n - kept
        naive = np.where(np.arange(n) < train_end, "train", np.where(np.arange(n) < val_end, "val", "test"))
        report.uploader_overlap_resolved = int(
            (df["_split"].to_numpy() != naive[: len(df)]).sum()) if len(df) == n else report.temporal_dropped
    out = []
    for part in (train_df, val_df, test_df):
        out.append(part.drop(columns=["_split"]).reset_index(drop=True))
    return out[0], out[1], out[2]

# ==============================================================================
# 8. FULL PIPELINE ORCHESTRATOR
# ==============================================================================
def run_pipeline(source: Union[str, Path, pd.DataFrame], output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CleaningReport]:
    report = CleaningReport()
    chunks = []
    
    for raw_chunk in load_raw_records(source):
        cleaned, report = clean_records(raw_chunk, report)
        if len(cleaned) > 0: chunks.append(cleaned)
        
    if not chunks: raise ValueError("No valid records after cleaning.")
    
    df = pd.concat(chunks, ignore_index=True)
    
    df = dedup_dataframe(df)
    report.duplicate_phash_removed = int(df["dup_phash"].sum())
    report.duplicate_embedding_removed = int(df["dup_embedding"].sum())
    report.duplicate_hashes_removed = int(df["is_duplicate"].sum())
    df = df[~df["is_duplicate"]].drop(columns=["is_duplicate", "dup_phash", "dup_embedding"]).reset_index(drop=True)

    train_df, val_df, test_df = temporal_group_split(df, report=report)
    
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(out / "train.parquet", index=False)
        val_df.to_parquet(out / "val.parquet", index=False)
        test_df.to_parquet(out / "test.parquet", index=False)
        (out / "cleaning_report.txt").write_text(report.summary(), encoding="utf-8")
        
    return train_df, val_df, test_df, report

# ==============================================================================
# 9. RIGOROUS & ADVERSARIAL UNIT TEST SUITE
# ==============================================================================
class TestWikimediaPipeline(unittest.TestCase):
    def test_1_maintenance_filtering_and_safelists(self):
        report = CleaningReport()
        cats = ["Nature", "Category:Maintenance categories", "Files uploaded with X", "Commons:Featured pictures"]
        clean = _clean_category_list(cats, report)
        self.assertEqual(clean, ["Nature", "Commons:Featured pictures"])
        self.assertEqual(report.maintenance_categories_removed, 2)

    def test_2_cleaning_null_safety(self):
        report = CleaningReport()
        df = pd.DataFrame({
            'categories': [["Nature"], None, '["Animals"]'], 
            'uploader': ["user1", "user1", "user2"], 
            'timestamp': ["2020-01-01", "invalid_date", "2020-01-02"] 
        })
        res, _ = clean_records(df, report)
        self.assertEqual(len(res), 2)
        self.assertEqual(res['categories'].iloc[1], ["Animals"]) 

    def test_3_uploader_grouped_split_no_leakage(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            'timestamp': dates,
            'uploader': ['u1']*3 + ['leaky_user']*2 + ['u2']*3 + ['leaky_user']*2,
            'categories': [['A']]*10
        })
        train, val, test = temporal_group_split(df, ratios=(0.5, 0.25, 0.25))
        self.assertTrue(set(train['uploader']).isdisjoint(set(val['uploader'])))
        self.assertTrue(set(train['uploader']).isdisjoint(set(test['uploader'])))
        self.assertTrue(set(val['uploader']).isdisjoint(set(test['uploader'])))

    def test_4_embedding_deduplication(self):
        emb1, emb2, emb3 = [1.0, 0.0], [0.99, 0.05], [0.0, 1.0]
        df = pd.DataFrame({'embeddings': [emb1, emb2, emb3], 'id': [1, 2, 3]})
        res = remove_near_duplicates_embeddings(df)
        res_filtered = res[~res['is_duplicate']]
        self.assertEqual(len(res_filtered), 2)

    def test_adversarial_maintenance_tags(self):
        report = CleaningReport()
        cats = [
            "category:files uploaded with upload wizard",
            "Category:MAINTENANCE_CLEANUP_2026",
            "Category:Unidentified_images",  
            "Category:  Hidden   Categories  ", 
            "Nature", 
            "Category:Speedy_Deletion_Requests"
        ]
        clean = _clean_category_list(cats, report)
        self.assertEqual(clean, ["Category:Unidentified_images", "Nature"])
        self.assertEqual(report.maintenance_categories_removed, 4)

    def test_malformed_inputs(self):
        report = CleaningReport()
        df = pd.DataFrame({
            'categories': [None, "", "   ", 123, 45.6, True, ["Valid"]],
            'uploader': ["u1"] * 7,
            'timestamp': ["2020-01-01"] * 7
        })
        res, _ = clean_records(df, report)
        self.assertEqual(len(res), 7)
        self.assertEqual(res['categories'].iloc[0], []) 
        self.assertEqual(res['categories'].iloc[6], ["Valid"]) 

    def test_data_leakage_and_duplicates(self):
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.99, 0.05, 0.0] 
        emb3 = [0.0, 1.0, 0.0]
        emb4 = [0.0, 0.0, 1.0]
        emb5 = [0.0, 0.0, 0.99] 
        df = pd.DataFrame({
            'timestamp': pd.date_range("2020-01-01", periods=5, freq="D"),
            'uploader': ['u1', 'u2', 'u3', 'u4', 'u5'],
            'categories': [['A'], ['B'], ['C'], ['D'], ['E']],
            'embeddings': [emb1, emb2, emb3, emb4, emb5]
        })
        df_deduped = remove_near_duplicates_embeddings(df)
        df_deduped = df_deduped[~df_deduped['is_duplicate']]
        self.assertEqual(len(df_deduped), 3) 
        train, val, test = temporal_group_split(df_deduped, ratios=(0.5, 0.25, 0.25))
        self.assertTrue(set(train['uploader']).isdisjoint(set(val['uploader'])))
        self.assertTrue(set(train['uploader']).isdisjoint(set(test['uploader'])))

    def test_temporal_boundaries(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        dates_list = list(dates)
        dates_list[6] = dates_list[7] 
        df = pd.DataFrame({
            'timestamp': dates_list,
            'uploader': [f'u{i}' for i in range(10)],
            'categories': [['A']]*10
        })
        train, val, test = temporal_group_split(df, ratios=(0.7, 0.1, 0.2))
        self.assertEqual(len(train) + len(val) + len(test), 10)

# FIX: Corrected the entry point so tests actually run!
if __name__ == "__main__":
    unittest.main(verbosity=2)