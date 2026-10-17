"""
extract_embeddings.py - build the retrieval artifacts from the cleaned splits.

Passes (each streams; nothing holds the whole dataset in Python lists):
  1. stream_encode   : images -> frozen-encoder embeddings, written batch by batch to a store
                       (Parquet via a pyarrow ParquetWriter in production). A failed image is skipped and
                       its labels are dropped WITH it (labels are tracked per loaded image).
  2. dedup_pass      : near-duplicate removal in embedding space (cosine), earliest upload survives.
                       Runs AFTER extraction because it needs the embeddings; pHash already ran before.
  3. build_index_pass: FAISS index (type chosen by size, logged) from the TRAIN split only, built with
                       one index.add per batch, plus row-aligned 64px thumbnails.
Outputs (see manifest.py): dataset_embeddings.parquet, faiss_index.index, index_labels.json,
neighbor_thumbs.npy, text_prototypes.npy, manifest.json (the only file that defines the class ids).
The encoder is injected, so the logic is testable without torch; labels come from the multi-label
``categories`` lists in the cleaned metadata, not from folder names (folder mode is kept as a fallback).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from cleaning_and_split import canonical_category_name, mark_near_duplicates_embeddings, normalize_category_name
from encoders import get_encoder, make_text_prototypes
from indexing import IndexBuilder, save_index
from manifest import DEFAULT_FILES, Manifest, write_manifest
from protocache import load_or_compute_prototypes

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("EmbeddingExtractor")

ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}
THUMB_SIZE = 64
META_SPLITS = ("val", "test")          # metadata text embeddings are only needed to fit/evaluate fusion


# ==============================================================================
# Records
# ==============================================================================
@dataclass
class Record:
    path: str
    categories: List[str]
    split: str = "train"
    uploader: str = ""
    timestamp: str = ""
    title: str = ""
    description: str = ""


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                return [str(x) for x in json.loads(s)]
            except json.JSONDecodeError:
                pass
        return [s] if s else []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(x) for x in value]
    return []


def _canon_list(cats: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for c in cats:
        name = canonical_category_name(c)
        key = normalize_category_name(name)
        if name and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _txt(v: Any) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def records_from_frames(frames: Dict[str, pd.DataFrame]) -> List[Record]:
    """Multi-label records from cleaned split frames ({'train': df, 'val': df, 'test': df})."""
    recs: List[Record] = []
    for split, df in frames.items():
        for row in df.to_dict("records"):
            recs.append(Record(
                path=str(row["image_path"]), categories=_canon_list(_as_list(row.get("categories"))), split=split,
                uploader=_txt(row.get("uploader")), timestamp=_txt(row.get("timestamp")),
                title=_txt(row.get("title")), description=_txt(row.get("description"))))
    return recs


def records_from_folders(data_dir: str) -> List[Record]:
    """Legacy fallback: data_dir/<CategoryName>/image.jpg -> one category per image, all in 'train'."""
    recs: List[Record] = []
    for root, _dirs, files in sorted(os.walk(data_dir)):
        for f in sorted(files):
            if Path(f).suffix.lower() in ALLOWED_EXTS:
                recs.append(Record(path=os.path.join(root, f), split="train",
                                   categories=_canon_list([os.path.basename(root)])))
    return recs


def load_splits_dir(splits_dir: str) -> Dict[str, pd.DataFrame]:
    out = {}
    for split in ("train", "val", "test"):
        p = Path(splits_dir) / f"{split}.parquet"
        if p.is_file():
            out[split] = pd.read_parquet(p)
    if "train" not in out:
        raise FileNotFoundError(f"{splits_dir}/train.parquet not found (run cleaning_and_split.py first)")
    return out


def build_label_space(records: Sequence[Record], min_freq: int = 20) -> Tuple[List[str], List[int]]:
    """Classes = categories with >= min_freq TRAIN images, most frequent first. list index == class id."""
    counts: Dict[str, int] = {}
    for r in records:
        if r.split == "train":
            for c in r.categories:
                counts[c] = counts.get(c, 0) + 1
    kept = sorted(((c, n) for c, n in counts.items() if n >= min_freq), key=lambda kv: (-kv[1], kv[0]))
    return [c for c, _ in kept], [n for _, n in kept]


# ==============================================================================
# Stores
# ==============================================================================
class MemoryStore:
    """In-memory store (tests, tiny datasets)."""

    def __init__(self, path: Any = None):
        self.rows: List[Dict[str, Any]] = []

    def append_batch(self, rows: List[Dict[str, Any]]) -> None:
        self.rows.extend(rows)

    def iter_batches(self, batch_size: int = 1024, columns: Optional[Sequence[str]] = None) -> Iterator[List[Dict[str, Any]]]:
        for s in range(0, len(self.rows), batch_size):
            chunk = self.rows[s:s + batch_size]
            yield [r if columns is None else {k: r[k] for k in columns} for r in chunk]

    def __len__(self) -> int:
        return len(self.rows)

    def close(self) -> None:
        pass


class ParquetStore:
    """Streaming Parquet store: pyarrow ParquetWriter, one row group per appended batch."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._writer = None
        self._dim: Optional[int] = None
        self._n = 0

    def _schema(self, dim: int):
        import pyarrow as pa
        return pa.schema([
            ("path", pa.string()), ("split", pa.string()), ("categories", pa.list_(pa.string())),
            ("uploader", pa.string()), ("timestamp", pa.string()), ("width", pa.int32()), ("height", pa.int32()),
            ("thumb", pa.binary()), ("embedding", pa.list_(pa.float32(), dim)),
            ("meta_embedding", pa.list_(pa.float32()))])

    def append_batch(self, rows: List[Dict[str, Any]]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if not rows:
            return
        dim = int(len(rows[0]["embedding"]))
        if self._writer is None:
            self._dim = dim
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(str(self.path), self._schema(dim))
        emb = pa.FixedSizeListArray.from_arrays(
            pa.array(np.concatenate([np.asarray(r["embedding"], np.float32) for r in rows]), pa.float32()), dim)
        meta = pa.array([[] if r.get("meta_embedding") is None else np.asarray(r["meta_embedding"], np.float32).tolist()
                         for r in rows], pa.list_(pa.float32()))
        table = pa.table({
            "path": pa.array([r["path"] for r in rows], pa.string()),
            "split": pa.array([r["split"] for r in rows], pa.string()),
            "categories": pa.array([list(r["categories"]) for r in rows], pa.list_(pa.string())),
            "uploader": pa.array([r.get("uploader", "") for r in rows], pa.string()),
            "timestamp": pa.array([r.get("timestamp", "") for r in rows], pa.string()),
            "width": pa.array([int(r.get("width", 0)) for r in rows], pa.int32()),
            "height": pa.array([int(r.get("height", 0)) for r in rows], pa.int32()),
            "thumb": pa.array([np.asarray(r["thumb"], np.uint8).tobytes() for r in rows], pa.binary()),
            "embedding": emb, "meta_embedding": meta}, schema=self._schema(dim))
        self._writer.write_table(table)
        self._n += len(rows)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __len__(self) -> int:
        return self._n

    def iter_batches(self, batch_size: int = 1024, columns: Optional[Sequence[str]] = None) -> Iterator[List[Dict[str, Any]]]:
        import pyarrow.parquet as pq
        self.close()
        if not self.path.is_file():
            return
        pf = pq.ParquetFile(str(self.path))
        for batch in pf.iter_batches(batch_size=batch_size, columns=list(columns) if columns else None):
            cols = batch.schema.names
            data: Dict[str, Any] = {}
            for name in cols:
                col = batch.column(name)
                if name == "embedding":
                    data[name] = col.values.to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)
                elif name == "thumb":
                    data[name] = [np.frombuffer(b, np.uint8).reshape(THUMB_SIZE, THUMB_SIZE, 3) for b in col.to_pylist()]
                elif name == "meta_embedding":
                    data[name] = [np.asarray(m, np.float32) if m else None for m in col.to_pylist()]
                else:
                    data[name] = col.to_pylist()
            yield [{name: data[name][i] for name in cols} for i in range(batch.num_rows)]


# ==============================================================================
# Pass 1: encode
# ==============================================================================
@dataclass
class ExtractStats:
    ok: int = 0
    failed: List[Tuple[str, str]] = field(default_factory=list)


def default_load_image(path: str) -> Tuple[Image.Image, int, int]:
    img = Image.open(path)
    img.load()                                             # force decoding so corrupt files fail HERE
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img, img.width, img.height


def make_thumb(img: Image.Image, size: int = THUMB_SIZE) -> np.ndarray:
    return np.asarray(img.resize((size, size)), dtype=np.uint8)


def stream_encode(records: Sequence[Record], encoder, store, batch_size: int = 32,
                  load_image: Callable[[str], Tuple[Image.Image, int, int]] = default_load_image) -> ExtractStats:
    """Encode records batch by batch. Only images that loaded are encoded, and each embedding is paired with the
    labels of ITS OWN record (the previous ``batch_paths[:len(valid_imgs)]`` slicing shifted labels after a failure)."""
    stats = ExtractStats()
    for s in range(0, len(records), batch_size):
        loaded: List[Tuple[Record, Image.Image, int, int]] = []
        for rec in records[s:s + batch_size]:
            try:
                img, w, h = load_image(rec.path)
                loaded.append((rec, img, w, h))
            except Exception as exc:
                logger.warning("skipping unreadable image %s: %s", rec.path, exc)
                stats.failed.append((rec.path, repr(exc)))
        if not loaded:
            continue
        emb = np.asarray(encoder.encode_images([x[1] for x in loaded]), dtype=np.float32)
        if len(emb) != len(loaded):
            raise RuntimeError(f"encoder returned {len(emb)} embeddings for {len(loaded)} images")
        meta_idx = [i for i, (rec, *_rest) in enumerate(loaded)
                    if rec.split in META_SPLITS and (rec.title or rec.description)]
        meta_emb: Dict[int, np.ndarray] = {}
        if meta_idx:
            texts = [f"{loaded[i][0].title}. {loaded[i][0].description}".strip(" .") for i in meta_idx]
            vec = make_text_prototypes(encoder, texts)
            meta_emb = {i: vec[k] for k, i in enumerate(meta_idx)}
        store.append_batch([{
            "path": rec.path, "split": rec.split, "categories": list(rec.categories), "uploader": rec.uploader,
            "timestamp": rec.timestamp, "width": w, "height": h, "thumb": make_thumb(img),
            "embedding": emb[i], "meta_embedding": meta_emb.get(i)} for i, (rec, img, w, h) in enumerate(loaded)])
        stats.ok += len(loaded)
        for _rec, img, _w, _h in loaded:
            img.close()
        if (s // batch_size) % 10 == 0:
            logger.info("encoded %d/%d images (%d failed)", stats.ok, len(records), len(stats.failed))
    return stats


# ==============================================================================
# Pass 2: embedding dedup
# ==============================================================================
def dedup_pass(src, dst, threshold: float = 0.95, block_size: int = 1024, batch_size: int = 1024) -> int:
    """Copy ``src`` to ``dst`` without near-duplicates (cosine >= threshold). The earliest upload survives, so a later
    copy in test never keeps its original out of train. Returns the number of rows removed."""
    embs, stamps = [], []
    for batch in src.iter_batches(batch_size, columns=["embedding", "timestamp"]):
        embs.extend(r["embedding"] for r in batch)
        stamps.extend(r["timestamp"] for r in batch)
    if not embs:
        return 0
    X = np.stack(embs).astype(np.float32)
    ts = pd.to_datetime(pd.Series(stamps), errors="coerce", utc=True)
    key = ts.astype("int64").where(ts.notna(), np.iinfo(np.int64).max).to_numpy()
    order = np.argsort(key, kind="stable")
    flags_sorted = mark_near_duplicates_embeddings(X[order], threshold=threshold, block_size=block_size)
    dup = np.zeros(len(X), dtype=bool)
    dup[order] = flags_sorted
    del X, embs
    i = 0
    for batch in src.iter_batches(batch_size):
        keep = [r for r, d in zip(batch, dup[i:i + len(batch)]) if not d]
        dst.append_batch(keep)
        i += len(batch)
    return int(dup.sum())


# ==============================================================================
# Pass 3: index (train only)
# ==============================================================================
def build_index_pass(store, class_to_id: Dict[str, int], backend: str = "auto",
                     thumbs_path: Optional[Union[str, Path]] = None, batch_size: int = 1024):
    """Index TRAIN rows that carry at least one in-vocabulary label. One ``index.add`` per batch.
    Returns (index, index_labels as class-id lists, thumbs array, index kind)."""
    def eligible(r) -> List[int]:
        return [class_to_id[c] for c in r["categories"] if c in class_to_id] if r["split"] == "train" else []

    n_idx = 0
    for batch in store.iter_batches(batch_size, columns=["split", "categories"]):
        n_idx += sum(1 for r in batch if eligible(r))
    if n_idx == 0:
        raise ValueError("no train images with in-vocabulary labels; lower --min_freq or check the splits")
    builder, labels, pos = None, [], 0
    thumbs = None
    for batch in store.iter_batches(batch_size):
        rows = [(r, eligible(r)) for r in batch]
        rows = [(r, ids) for r, ids in rows if ids]
        if not rows:
            continue
        emb = np.stack([r["embedding"] for r, _ in rows]).astype(np.float32)
        if builder is None:
            builder = IndexBuilder(emb.shape[1], n_idx, backend)
            shape = (n_idx, THUMB_SIZE, THUMB_SIZE, 3)
            thumbs = (np.lib.format.open_memmap(str(thumbs_path), mode="w+", dtype=np.uint8, shape=shape)
                      if thumbs_path else np.zeros(shape, np.uint8))
        builder.add(emb)                                              # per-batch add
        thumbs[pos:pos + len(rows)] = np.stack([np.asarray(r["thumb"], np.uint8) for r, _ in rows])
        labels.extend(ids for _, ids in rows)
        pos += len(rows)
    index = builder.finish()
    if hasattr(thumbs, "flush"):
        thumbs.flush()
    return index, labels, thumbs, builder.kind


# ==============================================================================
# Orchestration
# ==============================================================================
def build_artifacts(source: Union[Dict[str, pd.DataFrame], Sequence[Record]], out_dir: Union[str, Path], encoder,
                    min_freq: int = 20, batch_size: int = 32, backend: str = "auto",
                    store_factory: Optional[Callable[[Path], Any]] = None, graph_path: Optional[str] = None,
                    dedup_threshold: float = 0.95, keep_raw: bool = False) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store_factory = store_factory or ParquetStore
    records = records_from_frames(source) if isinstance(source, dict) else list(source)
    classes, counts = build_label_space(records, min_freq)
    if not classes:
        raise ValueError(f"no category has >= {min_freq} train images; lower --min_freq")
    class_to_id = {c: i for i, c in enumerate(classes)}
    logger.info("label space: %d classes (min_freq=%d) from %d records", len(classes), min_freq, len(records))

    raw = store_factory(out / "embeddings_raw.parquet")
    stats = stream_encode(records, encoder, raw, batch_size)
    raw.close()

    final = store_factory(out / DEFAULT_FILES["embeddings"])
    n_dup = dedup_pass(raw, final, threshold=dedup_threshold)
    final.close()
    if not keep_raw and isinstance(raw, ParquetStore):
        raw.path.unlink(missing_ok=True)

    index, id_labels, _thumbs, kind = build_index_pass(final, class_to_id, backend, thumbs_path=out / DEFAULT_FILES["thumbs"])
    save_index(index, out / DEFAULT_FILES["index"])
    (out / DEFAULT_FILES["index_labels"]).write_text(json.dumps(id_labels), encoding="utf-8")
    load_or_compute_prototypes(out, classes, lambda names: make_text_prototypes(encoder, names), encoder.model_name)
    if graph_path:
        shutil.copyfile(graph_path, out / DEFAULT_FILES["graph"])

    write_manifest(out, Manifest(
        model_name=encoder.model_name, embed_dim=int(encoder.embed_dim), classes=classes, class_counts=counts,
        index_type=kind, n_vectors=int(index.ntotal),
        logit_scale=float(encoder.logit_scale), logit_bias=float(encoder.logit_bias)), expected_dim=int(encoder.embed_dim))

    by_split: Dict[str, int] = {}
    for batch in final.iter_batches(2048, columns=["split"]):
        for r in batch:
            by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    summary = {"records": len(records), "rows_encoded": stats.ok, "failed": len(stats.failed),
               "duplicates_removed": n_dup, "rows_final": sum(by_split.values()), "rows_by_split": by_split,
               "n_index": int(index.ntotal), "index_type": kind, "classes": len(classes)}
    (out / "extraction_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("done: %s", summary)
    return summary


# ==============================================================================
# Self-test (no model, no network)
# ==============================================================================
def run_test() -> None:
    from encoders import FakeEncoder
    tmp = Path(tempfile.mkdtemp())
    try:
        rng = np.random.default_rng(0)
        cols = {"Reds": (230, 20, 20), "Greens": (20, 200, 20), "Blues": (20, 20, 230)}
        rows = []
        for cat, col in cols.items():
            for i in range(6):
                arr = np.clip(np.array(col) + rng.integers(-60, 60, size=(32, 32, 3)), 0, 255).astype("uint8")
                p = tmp / f"{cat}_{i}.png"
                Image.fromarray(arr).save(p)
                rows.append((str(p), [cat], f"2020-01-{len(rows) + 1:02d}"))
        (tmp / "bad.png").write_bytes(b"corrupt")
        rows.append((str(tmp / "bad.png"), ["Reds"], "2020-02-01"))
        df = pd.DataFrame({"image_path": [r[0] for r in rows], "categories": [r[1] for r in rows],
                           "timestamp": [r[2] for r in rows], "uploader": [f"u{i}" for i in range(len(rows))]})
        summary = build_artifacts({"train": df}, tmp / "artifacts", FakeEncoder(), min_freq=2, batch_size=4,
                                  store_factory=MemoryStore, backend="numpy", dedup_threshold=1.01)
        assert summary["failed"] == 1 and summary["n_index"] == 18, summary
        print("EXTRACTION SELF-TEST PASSED:", summary)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract embeddings and build retrieval artifacts.")
    ap.add_argument("--splits_dir", help="Directory with train/val/test.parquet from cleaning_and_split.py")
    ap.add_argument("--data_dir", help="LEGACY: folder-per-category images (single label, all 'train')")
    ap.add_argument("--output_dir", default="./artifacts")
    ap.add_argument("--min_freq", type=int, default=20, help="Min train images per class (PDF: 20-50 for the head)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--encoder", default=None, help="'siglip' (default) or 'fake' (tests/CI only)")
    ap.add_argument("--graph", default=None, help="category_graph.json to ship with the artifacts")
    ap.add_argument("--dedup_threshold", type=float, default=0.95)
    ap.add_argument("--test", action="store_true", help="Run the self-test (no model download)")
    args = ap.parse_args()
    if args.test:
        run_test()
        return
    if args.splits_dir:
        source: Any = load_splits_dir(args.splits_dir)
    elif args.data_dir:
        source = records_from_folders(args.data_dir)
    else:
        ap.error("give --splits_dir (preferred) or --data_dir")
    build_artifacts(source, args.output_dir, get_encoder(args.encoder), min_freq=args.min_freq,
                    batch_size=args.batch_size, graph_path=args.graph, dedup_threshold=args.dedup_threshold)


if __name__ == "__main__":
    main()
