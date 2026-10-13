"""indexing.py - vector index selection and a FAISS-free fallback.

Policy (bug 14): exact Flat inner-product below FLAT_MAX_VECTORS; HNSW above that; IVF-PQ for very
large sets (about 64 B/vector, the figure in the problem statement). The choice is logged. When
faiss is not installed only Flat is available (numpy backend); asking for an ANN index then raises,
rather than silently running an O(N) scan over millions of vectors.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:          # ImportError, or a broken binary wheel
    faiss = None
    HAVE_FAISS = False

logger = logging.getLogger("CommonsIndex")

FLAT_MAX_VECTORS = 200_000
HNSW_MAX_VECTORS = 5_000_000


def choose_index_type(n_vectors: int) -> str:
    if n_vectors < FLAT_MAX_VECTORS:
        return "flat"
    if n_vectors < HNSW_MAX_VECTORS:
        return "hnsw"
    return "ivfpq"


class NumpyFlatIndex:
    """Exact inner-product index with the subset of the faiss API this project uses."""

    def __init__(self, dim: int):
        self.d = dim
        self._chunks: List[np.ndarray] = []
        self._mat: Optional[np.ndarray] = None

    @property
    def ntotal(self) -> int:
        return sum(len(c) for c in self._chunks)

    def _matrix(self) -> np.ndarray:
        if self._mat is None or len(self._mat) != self.ntotal:
            self._mat = np.concatenate(self._chunks) if self._chunks else np.zeros((0, self.d), np.float32)
            self._chunks = [self._mat]
        return self._mat

    def add(self, x: np.ndarray) -> None:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"expected (n, {self.d}) array, got {x.shape}")
        self._chunks.append(x.copy())

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(q, dtype=np.float32)
        M = self._matrix()
        n = len(M)
        D = np.full((len(q), k), -np.inf, dtype=np.float32)
        I = np.full((len(q), k), -1, dtype=np.int64)
        if n == 0:
            return D, I
        kk = min(k, n)
        sims = q @ M.T
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        rows = np.arange(len(q))[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        top = part[rows, order]
        I[:, :kk] = top
        D[:, :kk] = sims[rows, top]
        return D, I

    def reconstruct(self, i: int) -> np.ndarray:
        return self._matrix()[i].copy()


AnyIndex = Union[NumpyFlatIndex, "faiss.Index"]


def save_index(index: AnyIndex, path: Union[str, Path]) -> None:
    path = Path(path)
    if isinstance(index, NumpyFlatIndex):
        with open(path, "wb") as fh:
            np.savez(fh, vectors=index._matrix())
    else:
        faiss.write_index(index, str(path))


def load_index(path: Union[str, Path]) -> AnyIndex:
    path = Path(path)
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"PK":                                    # numpy .npz written by save_index
        with np.load(path) as z:
            vec = z["vectors"]
        idx = NumpyFlatIndex(vec.shape[1] if vec.ndim == 2 and vec.size else 0)
        if vec.size:
            idx.add(vec)
        return idx
    if not HAVE_FAISS:
        raise RuntimeError(f"{path} is a faiss index but faiss is not installed")
    return faiss.read_index(str(path))


def _pq_m(dim: int) -> int:
    for m in (64, 48, 32, 24, 16, 8, 4, 2, 1):
        if dim % m == 0:
            return m
    return 1


class IndexBuilder:
    """Streaming builder: ``add`` batches as they are produced, ``finish`` returns the index.

    The type is chosen once from ``expected_n`` and logged. IVF-PQ is trained on the first
    ``train_size`` vectors (buffered), then everything is added.
    """

    def __init__(self, dim: int, expected_n: int, backend: str = "auto"):
        self.dim = dim
        self.expected_n = max(int(expected_n), 1)
        self.kind = choose_index_type(self.expected_n)
        use_faiss = HAVE_FAISS and backend != "numpy"
        if self.kind != "flat" and not use_faiss:
            raise RuntimeError(f"{self.expected_n} vectors need an ANN index ({self.kind}) but faiss is unavailable")
        self._buffer: List[np.ndarray] = []
        self._trained = True
        self._train_size = 0
        if not use_faiss:
            self.index: AnyIndex = NumpyFlatIndex(dim)
            self.backend = "numpy"
        else:
            self.backend = "faiss"
            if self.kind == "flat":
                self.index = faiss.IndexFlatIP(dim)
            elif self.kind == "hnsw":
                self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 64
            else:
                nlist = int(min(max(4 * math.sqrt(self.expected_n), 64), 65536))
                quantizer = faiss.IndexFlatIP(dim)
                self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, _pq_m(dim), 8, faiss.METRIC_INNER_PRODUCT)
                self._trained = False
                self._train_size = int(min(self.expected_n, max(100_000, 40 * nlist)))
        logger.info("Index type chosen: %s (backend=%s, expected_n=%d, dim=%d)",
                    self.kind, self.backend, self.expected_n, dim)

    def add(self, x: np.ndarray) -> None:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self._trained:
            self.index.add(x)
            return
        self._buffer.append(x)
        if sum(len(b) for b in self._buffer) >= self._train_size:
            self._train_and_flush()

    def _train_and_flush(self) -> None:
        data = np.concatenate(self._buffer)
        self.index.train(data)
        self.index.add(data)
        self._buffer, self._trained = [], True

    def finish(self) -> AnyIndex:
        if not self._trained and self._buffer:
            self._train_and_flush()
        if self.kind == "ivfpq" and self.backend == "faiss":
            self.index.nprobe = 16
        logger.info("Index built: %s with %d vectors", self.kind, self.index.ntotal)
        return self.index
