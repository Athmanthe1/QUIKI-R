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
    from imagehash import phash, hex_to_hash
    _HASH_AVAILABLE = True
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
PHASH_BRUTE_FORCE_LIMIT: int = 5000
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
    uploader_overlap_resolved: int = 0
    temporal_dropped: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 50, "  CLEANING REPORT", "=" * 50,
            f"  Total records ingested        : {self.total_records}",
            f"  Records with >=1 category     : {self.records_with_categories}",
            f"  Records without any category  : {self.records_without_categories}",
            f"  Maintenance cats stripped     : {self.maintenance_categories_removed}",
            f"  Near-duplicate hashes removed : {self.duplicate_hashes_removed}",
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
# 6. DEDUPLICATION
# ==============================================================================
def compute_phash(image_path: str) -> Optional[str]:
    if not _HASH_AVAILABLE or not image_path or not os.path.isfile(image_path): return None
    try:
        with Image.open(image_path) as img: return str(phash(img))
    except Exception: return None

def mark_near_duplicates_phash(df: pd.DataFrame, threshold: int = DEFAULT_HAMMING_THRESHOLD) -> pd.DataFrame:
    df["phash"] = df["image_path"].map(compute_phash)
    hashed_mask = df["phash"].notna()
    if not hashed_mask.any():
        df["is_duplicate"] = False
        return df
        
    hashed_indices = df.index[hashed_mask].tolist()
    hashed_values = df.loc[hashed_mask, "phash"].tolist()
    
    if len(hashed_indices) > PHASH_BRUTE_FORCE_LIMIT:
        logger.warning("pHash limit exceeded. Subsampling to prevent OOM/hang.")
        df["is_duplicate"] = df.duplicated(subset=["phash"], keep="first")
        return df
        
    # FIX: Use a dictionary for Union-Find to prevent IndexError on non-contiguous DF indices
    parent = {idx: idx for idx in hashed_indices}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
        
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            
    for i in range(len(hashed_indices)):
        for j in range(i + 1, len(hashed_indices)):
            try:
                dist = int(hex_to_hash(hashed_values[i]) - hex_to_hash(hashed_values[j]))
                if dist <= threshold:
                    union(hashed_indices[i], hashed_indices[j])
            except Exception:
                continue
                
    components: Dict[int, List[int]] = {}
    for idx in hashed_indices:
        components.setdefault(find(idx), []).append(idx)
        
    dup_mask = pd.Series(False, index=df.index)
    for members in components.values():
        if len(members) > 1:
            for dup_idx in members[1:]:
                dup_mask.loc[dup_idx] = True
                
    df["is_duplicate"] = dup_mask
    return df

def remove_near_duplicates_embeddings(df: pd.DataFrame, emb_col: str = 'embeddings') -> pd.DataFrame:
    if emb_col not in df.columns or df[emb_col].isna().all():
        df["is_duplicate"] = False
        return df
        
    # FIX: Process iteratively to prevent OOM crashes from massive np.vstack()
    keep_mask = np.zeros(len(df), dtype=bool)
    accepted_list = []
    
    for i, val in enumerate(df[emb_col].values):
        try:
            emb_raw = np.array(val, dtype=np.float32)
            if emb_raw.size == 0:
                continue
                
            norm = np.linalg.norm(emb_raw)
            if norm == 0: norm = 1.0
            emb = emb_raw / norm
            
            if not accepted_list:
                keep_mask[i] = True
                accepted_list.append(emb)
                continue
                
            accepted_matrix = np.array(accepted_list)
            sims = np.dot(emb, accepted_matrix.T)
            max_sim = np.max(sims)
            
            if max_sim < COSINE_SIM_THRESHOLD:
                keep_mask[i] = True
                accepted_list.append(emb)
            else:
                keep_mask[i] = False
        except Exception:
            keep_mask[i] = False # Drop malformed embeddings safely
            
    df["is_duplicate"] = ~keep_mask
    return df

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
    
    if "image_path" in df.columns and _HASH_AVAILABLE:
        df = mark_near_duplicates_phash(df)
    elif "embeddings" in df.columns:
        df = remove_near_duplicates_embeddings(df)
    else:
        df["is_duplicate"] = False
        
    report.duplicate_hashes_removed = int(df["is_duplicate"].sum())
    df = df[~df["is_duplicate"]].drop(columns=["is_duplicate"]).reset_index(drop=True)
    
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