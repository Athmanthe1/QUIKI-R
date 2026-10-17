"""feedback.py - editor feedback storage, session overlay index and training-batch helpers (torch-free).

Feedback is stored PER IMAGE as {embedding, positives, negatives}; a later click on the same
category overrides the earlier one. Training targets are three-state: +1 positive, -1 explicit
reject (hard negative), 0 unknown. Accepted images are also appended to a per-session overlay index
so they influence kNN immediately, without retraining (PDF: "index updates ... no retraining").
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

from indexing import NumpyFlatIndex

logger = logging.getLogger("CommonsFeedback")

POS, UNK, NEG = 1.0, 0.0, -1.0


def image_id_from_embedding(emb: np.ndarray) -> str:
    return hashlib.sha1(np.round(np.asarray(emb, dtype=np.float32), 4).tobytes()).hexdigest()[:16]


@dataclass
class ImageFeedback:
    embedding: np.ndarray
    positives: Set[str] = field(default_factory=set)
    negatives: Set[str] = field(default_factory=set)


class FeedbackStore:
    def __init__(self) -> None:
        self.records: Dict[str, ImageFeedback] = {}

    def record(self, image_id: str, embedding: np.ndarray, category: str, accepted: bool) -> None:
        rec = self.records.get(image_id)
        if rec is None:
            rec = self.records[image_id] = ImageFeedback(np.asarray(embedding, dtype=np.float32))
        if accepted:
            rec.positives.add(category); rec.negatives.discard(category)
        else:
            rec.negatives.add(category); rec.positives.discard(category)

    def __len__(self) -> int:
        return len(self.records)

    def counts(self) -> Dict[str, int]:
        return {"images": len(self.records),
                "positives": sum(len(r.positives) for r in self.records.values()),
                "negatives": sum(len(r.negatives) for r in self.records.values())}

    def to_training_arrays(self, class_to_id: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """One row per image. Categories outside ``class_to_id`` are ignored."""
        if not self.records:
            return np.zeros((0, 0), "float32"), np.zeros((0, len(class_to_id)), "float32")
        X = np.stack([r.embedding for r in self.records.values()]).astype("float32")
        T = np.full((len(X), len(class_to_id)), UNK, dtype="float32")
        for i, r in enumerate(self.records.values()):
            for c in r.positives:
                if c in class_to_id: T[i, class_to_id[c]] = POS
            for c in r.negatives:
                if c in class_to_id: T[i, class_to_id[c]] = NEG
        return X, T


# ---------------- append-only JSONL persistence ----------------

def append_event(path: Union[str, Path], image_id: str, embedding: np.ndarray, category: str, accepted: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": time.time(), "image_id": image_id, "category": category, "accepted": bool(accepted),
                       "embedding": [round(float(x), 5) for x in np.asarray(embedding).ravel()]})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def load_events(path: Union[str, Path]) -> Tuple[FeedbackStore, int]:
    """Replay a JSONL log in order. Returns (store, number_of_unreadable_lines_skipped)."""
    store, skipped = FeedbackStore(), 0
    path = Path(path)
    if not path.is_file():
        return store, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
            store.record(ev["image_id"], np.asarray(ev["embedding"], dtype=np.float32), ev["category"], bool(ev["accepted"]))
        except Exception:
            skipped += 1
    return store, skipped


# ---------------- per-session overlay index ----------------

class SessionOverlay:
    """Small exact index of accepted images for ONE session (kNN picks it up immediately)."""

    def __init__(self, dim: int):
        self.index = NumpyFlatIndex(dim)
        self._row: Dict[str, int] = {}
        self._labels: List[Set[str]] = []
        self._thumbs: Dict[int, np.ndarray] = {}

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def accept(self, image_id: str, embedding: np.ndarray, category: str, thumb: Optional[np.ndarray] = None) -> None:
        if image_id not in self._row:
            self._row[image_id] = self.index.ntotal
            self.index.add(np.asarray(embedding, dtype=np.float32)[None, :])
            self._labels.append(set())
            if thumb is not None:
                self._thumbs[self._row[image_id]] = np.asarray(thumb, dtype=np.uint8)
        self._labels[self._row[image_id]].add(category)

    def reject(self, image_id: str, category: str) -> None:
        if image_id in self._row:
            self._labels[self._row[image_id]].discard(category)

    def thumb_of_row(self, row: int) -> Optional[np.ndarray]:
        return self._thumbs.get(row)

    def labels_of(self, image_id: str) -> List[str]:
        return sorted(self._labels[self._row[image_id]]) if image_id in self._row else []

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, Callable[[int], Optional[List[str]]]]:
        D, I = self.index.search(q, k)

        def lookup(row: int) -> Optional[List[str]]:
            if row < 0 or row >= len(self._labels) or not self._labels[row]:
                return None                                   # empty-label rows must not vote
            return sorted(self._labels[row])
        return D, I, lookup


# ---------------- training helpers ----------------

def iterate_minibatches(X: np.ndarray, T: np.ndarray, batch_size: int, rng: np.random.Generator) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    order = rng.permutation(len(X))
    for s in range(0, len(X), max(int(batch_size), 1)):
        idx = order[s:s + batch_size]
        yield X[idx], T[idx]


def replay_sample(offline_X: Optional[np.ndarray], offline_T: Optional[np.ndarray], n: int,
                  rng: np.random.Generator) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Random rows of the offline training data, mixed into fine-tuning to limit forgetting."""
    if offline_X is None or len(offline_X) == 0 or n <= 0:
        return None, None
    idx = rng.choice(len(offline_X), size=min(n, len(offline_X)), replace=False)
    return offline_X[idx], offline_T[idx]


def f1_labelled(probs: np.ndarray, T: np.ndarray, threshold: float = 0.5) -> float:
    """Micro F1 over LABELLED entries only (positives and explicit rejects); unknown entries are ignored."""
    pred = probs >= threshold
    tp = float((pred & (T > 0.5)).sum())
    fp = float((pred & (T < -0.5)).sum())
    fn = float((~pred & (T > 0.5)).sum())
    if tp == 0:
        return 0.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def holdout_split(n: int, frac: float = 0.2, seed: int = 0, min_images: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Image-level split. Below ``min_images`` there is no hold-out (callers must then report in-sample F1)."""
    idx = np.random.default_rng(seed).permutation(n)
    if n < min_images:
        return idx, idx[:0]
    n_ho = max(1, int(round(n * frac)))
    return idx[n_ho:], idx[:n_ho]
