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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
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

# Use lookarounds to allow underscores, hyphens, and spaces as separators.
# (?<![a-zA-Z0-9]) ensures we don't match partial words (e.g. "xmaintenance")
# (?![a-zA-Z0-9]) ensures we don't match partial words (e.g. "maintenancex")
# This perfectly handles Wikimedia's underscore-separated category names!
_BOUNDARY_START = r"(?<![a-zA-Z0-9])"
_BOUNDARY_END = r"(?![a-zA-Z0-9])"
_SEP = r"[\s_-]+"

MAINTENANCE_PATTERNS: List[re.Pattern] = [
    re.compile(f"{_BOUNDARY_START}maintenance{_BOUNDARY_END}", re.I),
    re.compile(f"{_BOUNDARY_START}cleanup{_BOUNDARY_END}|{_BOUNDARY_START}clean-up{_BOUNDARY_END}", re.I),
    re.compile(f"{_BOUNDARY_START}backlog{_BOUNDARY_END}", re.I),
    re.compile(f"{_BOUNDARY_START}tracking{_SEP}categories{_BOUNDARY_END}", re.I),
    re.compile(f"{_BOUNDARY_START}hidden{_SEP}categories{_BOUNDARY_END}", re.I),
    re.compile(f"{_BOUNDARY_START}process{_SEP}categories{_BOUNDARY_END}", re.I),
    re.compile(f"{_BOUNDARY_START}empty{_SEP}(categories|pages|files|galleries){_BOUNDARY_END}", re.I),
    re.compile(f"{_BOUNDARY_START}deleted{_SEP}(pages|files|revisions){_BOUNDARY_END}", re.I),
    re.compile(f"{_BOUNDARY_START}admin(istration)?{_SEP}(categories|pages|noticeboards?){_BOUNDARY_END}", re.I),
    re.compile(f"files?{_SEP}uploaded{_SEP}with{_BOUNDARY_END}", re.I),
    re.compile(f"files?{_SEP}with{_SEP}(broken|missing|no){_BOUNDARY_END}", re.I),
    re.compile(f"{_BOUNDARY_START}license{_SEP}templates?{_BOUNDARY_END}", re.I),
    re.compile(f"copyright{_SEP}violations?{_BOUNDARY_END}", re.I),
    re.compile(f"{_BOUNDARY_START}no{_SEP}(license|author|source|description|permission){_BOUNDARY_END}", re.I),
    re.compile(f"unfree{_SEP}(files?|media){_BOUNDARY_END}", re.I),
    re.compile(f"{_BOUNDARY_START}speedy{_SEP}deletion{_BOUNDARY_END}", re.I),
]

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

    def summary(self) -> str:
        lines = [
            "=" * 50, "  CLEANING REPORT", "=" * 50,
            f"  Total records ingested        : {self.total_records}",
            f"  Records with >=1 category     : {self.records_with_categories}",
            f"  Records without any category  : {self.records_without_categories}",
            f"  Maintenance cats stripped     : {self.maintenance_categories_removed}",
            f"  Near-duplicate hashes removed : {self.duplicate_hashes_removed}",
            f"  Records reassigned (grouping) : {self.uploader_overlap_resolved}",
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

def _is_maintenance_category(category: str) -> bool:
    if not category: return False
    bare = category
    for prefix in ("Category:", "category:", "CATEGORY:"):
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
            break
    bare = bare.strip()
    if any(bare.startswith(safe) for safe in CONTENT_SAFELIST_PREFIXES):
        return False
    return any(pat.search(bare) for pat in MAINTENANCE_PATTERNS)

def _clean_category_list(categories: Sequence[str], report: CleaningReport) -> List[str]:
    if not categories: return []
    seen, cleaned = set(), []
    for raw in categories:
        cat = _safe_str(raw)
        if not cat: continue
        if _is_maintenance_category(cat):
            report.maintenance_categories_removed += 1
            continue
        if cat not in seen:
            seen.add(cat)
            cleaned.append(cat)
    return cleaned

def clean_records(raw_df: pd.DataFrame, report: CleaningReport,
                  category_col: str = "categories", timestamp_col: str = "timestamp",
                  uploader_col: str = "uploader") -> Tuple[pd.DataFrame, CleaningReport]:
    if raw_df is None or len(raw_df) == 0:
        return pd.DataFrame(), report
        
    df = raw_df.copy()
    report.total_records += len(df)
    
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
        return _clean_category_list(list(cats), report)
        
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
    df = df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    n = len(df)
    train_end, val_end = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    
    uploader_first_ts = df.groupby(uploader_col)[timestamp_col].min().sort_values(kind="mergesort")
    uploader_counts = df.groupby(uploader_col).size()
    
    split_map, cumulative = {}, 0
    for uploader in uploader_first_ts.index:
        if cumulative < train_end: split_map[uploader] = "train"
        elif cumulative < val_end: split_map[uploader] = "val"
        else: split_map[uploader] = "test"
        cumulative += int(uploader_counts[uploader])
        
    df = df.assign(_split=df[uploader_col].map(split_map))
    
    if report:
        naive = np.where(np.arange(n) < train_end, "train", np.where(np.arange(n) < val_end, "val", "test"))
        report.uploader_overlap_resolved = int((df["_split"].to_numpy() != naive).sum())
        
    train_df = df[df["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    val_df = df[df["_split"] == "val"].drop(columns=["_split"]).reset_index(drop=True)
    test_df = df[df["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)
    
    return train_df, val_df, test_df

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