"""Bug 14: Flat below ~200k vectors, auto-switch to HNSW / IVF-PQ above, log the choice; cache text prototypes."""
import logging

import numpy as np
import pytest

import indexing
from indexing import (FLAT_MAX_VECTORS, IndexBuilder, NumpyFlatIndex, choose_index_type, load_index, save_index)
from protocache import load_or_compute_prototypes


def test_choice_thresholds():
    assert choose_index_type(1) == "flat"
    assert choose_index_type(FLAT_MAX_VECTORS - 1) == "flat"
    assert choose_index_type(FLAT_MAX_VECTORS) == "hnsw"
    assert choose_index_type(2_000_000) == "hnsw"
    assert choose_index_type(20_000_000) == "ivfpq"


def test_builder_logs_choice_and_small_set_is_exact(caplog):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 32)).astype("float32")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    with caplog.at_level(logging.INFO, logger="CommonsIndex"):
        b = IndexBuilder(dim=32, expected_n=len(X), backend="numpy")
        for i in range(0, 500, 100):
            b.add(X[i:i + 100])
        idx = b.finish()
    assert any("flat" in r.message for r in caplog.records)
    assert idx.ntotal == 500 and b.kind == "flat"
    D, I = idx.search(X[:5], 3)
    assert (I[:, 0] == np.arange(5)).all()                                   # self is nearest
    brute = np.argsort(-(X[:5] @ X.T), axis=1)[:, :3]
    assert (I == brute).all()


def test_numpy_index_roundtrip_and_k_larger_than_n(tmp_path):
    X = np.eye(4, dtype="float32")
    idx = NumpyFlatIndex(4)
    idx.add(X)
    save_index(idx, tmp_path / "i.index")
    got = load_index(tmp_path / "i.index")
    D, I = got.search(X[:1], 10)
    assert got.ntotal == 4 and I.shape == (1, 10) and I[0, 0] == 0 and (I[0, 4:] == -1).all()


def test_ann_without_faiss_fails_loudly_not_silently():
    if indexing.HAVE_FAISS:
        pytest.skip("faiss present")
    with pytest.raises(RuntimeError, match="faiss"):
        IndexBuilder(dim=8, expected_n=FLAT_MAX_VECTORS + 1, backend="auto")


def test_prototype_cache_hits_and_invalidates(tmp_path):
    calls = []

    def compute(names):
        calls.append(list(names))
        return np.arange(len(names) * 4, dtype="float32").reshape(len(names), 4)

    a = load_or_compute_prototypes(tmp_path, ["A", "B"], compute, model_name="m")
    b = load_or_compute_prototypes(tmp_path, ["A", "B"], compute, model_name="m")
    assert len(calls) == 1 and (a == b).all()                                # second call served from disk
    load_or_compute_prototypes(tmp_path, ["A", "B", "C"], compute, model_name="m")   # class list changed
    load_or_compute_prototypes(tmp_path, ["A", "B", "C"], compute, model_name="other")   # model changed
    assert len(calls) == 3


def test_prototype_cache_incremental_for_new_categories(tmp_path):
    calls = []

    def compute(names):
        calls.append(list(names))
        return np.ones((len(names), 4), dtype="float32")

    load_or_compute_prototypes(tmp_path, ["A", "B"], compute, model_name="m")
    out = load_or_compute_prototypes(tmp_path, ["A", "B", "C"], compute, model_name="m", incremental=True)
    assert calls[-1] == ["C"] and out.shape == (3, 4)
