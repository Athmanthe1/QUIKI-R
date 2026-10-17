"""manifest.py - the single artifact contract between extraction, training and the engine.

One file, ``manifest.json``, written by exactly ONE script (extract_embeddings.py) and read by
the engine and the trainer. The ordered ``classes`` list defines the class id space:
**list index == class id**, everywhere (index labels, head output rows, engine).

This module is deliberately torch-free so the contract can be tested and used anywhere.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
SIGLIP_EMBED_DIM = 768
MODEL_NAME = SIGLIP_MODEL_NAME

# Canonical artifact file names (relative to the artifact directory).
DEFAULT_FILES: Dict[str, str] = {
    "index": "faiss_index.index",              # FAISS index over TRAIN images only
    "index_labels": "index_labels.json",       # row -> list of class ids (ints)
    "head": "supervised_head.npz",             # linear head: W [C, D], b [C]  (written by trainer.py)
    "graph": "category_graph.json",            # category DAG: {"edges": [[parent, child], ...]}
    "thumbs": "neighbor_thumbs.npy",           # sidecar: uint8 [N, 64, 64, 3] thumbnails, row-aligned with the index
    "stacker": "stacker.json",                 # logistic-regression fusion (written by fit_stacker)
    "text_prototypes": "text_prototypes.npy",  # cached [C, D] text prototypes
    "embeddings": "dataset_embeddings.parquet",
}


class ManifestError(ValueError):
    """Raised for a missing, malformed or incompatible manifest."""


@dataclass
class Manifest:
    model_name: str
    embed_dim: int
    classes: List[str]
    schema_version: int = SCHEMA_VERSION
    artifact_version: str = "1.4.0"
    index_type: str = "flat"
    n_vectors: int = 0
    files: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FILES))
    class_counts: Optional[List[int]] = None       # train images per class, aligned with `classes`
    logit_scale: Optional[float] = None            # SigLIP log-scale (text scorer: sigmoid(exp(s) * z.t + b))
    logit_bias: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    notes: str = ""

    @property
    def class_to_id(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}

    @property
    def id_to_class(self) -> Dict[int, str]:
        return dict(enumerate(self.classes))

    def path(self, directory: Union[str, Path], key: str) -> Path:
        return Path(directory) / self.files[key]


def _validate(m: Manifest, expected_dim: Optional[int] = None) -> None:
    if not isinstance(m.classes, list) or not all(isinstance(c, str) and c for c in m.classes):
        raise ManifestError("manifest.classes must be a list of non-empty strings")
    if len(set(m.classes)) != len(m.classes):
        dup = sorted({c for c in m.classes if m.classes.count(c) > 1})
        raise ManifestError(f"manifest.classes contains duplicates: {dup[:5]}")
    if not isinstance(m.embed_dim, int) or m.embed_dim <= 0:
        raise ManifestError(f"invalid embed_dim: {m.embed_dim!r}")
    if expected_dim is not None and m.embed_dim != expected_dim:
        raise ManifestError(f"embed_dim {m.embed_dim} != expected {expected_dim}")
    if m.class_counts is not None and len(m.class_counts) != len(m.classes):
        raise ManifestError("class_counts must align with classes")


def write_manifest(directory: Union[str, Path], manifest: Manifest, expected_dim: Optional[int] = None) -> Path:
    """Validate and atomically write ``manifest.json`` into ``directory``."""
    _validate(manifest, expected_dim)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MANIFEST_NAME
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".manifest.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(asdict(manifest), fh, indent=2, ensure_ascii=False)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


def read_manifest(directory: Union[str, Path], expected_dim: Optional[int] = None) -> Manifest:
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        raise ManifestError(f"{MANIFEST_NAME} not found in {directory}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path} is not valid JSON: {exc}") from exc
    if int(raw.get("schema_version", 0)) > SCHEMA_VERSION:
        raise ManifestError(f"manifest schema {raw.get('schema_version')} is newer than supported {SCHEMA_VERSION}")
    known = {k: raw[k] for k in Manifest.__dataclass_fields__ if k in raw}
    missing = {"model_name", "embed_dim", "classes"} - set(known)
    if missing:
        raise ManifestError(f"manifest missing required keys: {sorted(missing)}")
    files = dict(DEFAULT_FILES)
    files.update(known.pop("files", {}) or {})
    m = Manifest(files=files, **known)
    _validate(m, expected_dim)
    return m


def save_head_npz(path: Union[str, Path], W: "np.ndarray", b: "np.ndarray") -> None:
    """Linear head as plain arrays (W [C, D], b [C]); readable without torch."""
    np.savez(path, W=np.asarray(W, dtype="float32"), b=np.asarray(b, dtype="float32"))


def load_head_npz(path: Union[str, Path]):
    with np.load(path) as z:
        return z["W"].astype("float32"), z["b"].astype("float32")
