"""Bug 5: dedup must be dual-mode, scale past 5000 rows without exact-match fallback, and not be O(n^2) python."""
import time

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import cleaning_and_split as cs


def _hex(h):
    return format(int(h), "016x")


def test_hash_dedup_finds_near_duplicates_beyond_5000_rows():
    rng = np.random.default_rng(0)
    n = 6000                                                     # > the old PHASH_BRUTE_FORCE_LIMIT
    hashes = rng.integers(0, 2**63, size=n, dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, size=n, dtype=np.uint64)
    planted = {}
    for src in range(0, 100, 2):                                 # 50 planted near-duplicates (1-6 flipped bits)
        h = int(hashes[src])
        for b in rng.choice(64, size=int(rng.integers(1, 7)), replace=False):
            h ^= 1 << int(b)
        hashes[src + 1] = np.uint64(h)
        planted[src + 1] = src
    df = pd.DataFrame({"phash": [_hex(h) for h in hashes]})
    out = cs.mark_near_duplicates_hashes(df, threshold=8)
    flagged = set(np.flatnonzero(out["is_duplicate"].to_numpy()))
    assert set(planted) <= flagged                               # exact-match fallback would find none of these
    assert len(flagged - set(planted)) <= 2                      # random 64-bit hashes are ~32 bits apart


def test_hash_dedup_threshold_is_respected():
    a = 0xF0F0F0F0F0F0F0F0
    df = pd.DataFrame({"phash": [_hex(a), _hex(a ^ 0b111), _hex(a ^ 0xFFFF)]})   # distance 3 and 16
    flags = cs.mark_near_duplicates_hashes(df, threshold=8)["is_duplicate"].tolist()
    assert flags == [False, True, False]


def test_embedding_dedup_matches_naive_greedy_across_block_boundaries():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 16)).astype("float32")
    for i in range(1, 300, 7):                                   # plant near-duplicates of earlier rows
        X[i] = X[i - 1] + rng.normal(scale=0.01, size=16)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    kept, naive = [], np.zeros(300, bool)
    for i in range(300):                                         # reference implementation
        if not kept or float(np.max(Xn[kept] @ Xn[i])) < 0.95:
            kept.append(i)
            naive[i] = True
    got = ~cs.mark_near_duplicates_embeddings(X, threshold=0.95, block_size=32)
    assert (got == naive).all()


def test_embedding_dedup_scales():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20000, 32)).astype("float32")
    t = time.time()
    dup = cs.mark_near_duplicates_embeddings(X, threshold=0.95, block_size=1024)
    assert time.time() - t < 60
    assert dup.sum() < 50


def test_pipeline_runs_both_dedup_modes(tmp_path):
    imgs = []
    rng = np.random.default_rng(3)
    base = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    other = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    for name, arr in [("a.png", base), ("a_copy.png", np.clip(base.astype(int) + 2, 0, 255).astype("uint8")),
                      ("b.png", other), ("c.png", rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))]:
        Image.fromarray(arr).save(tmp_path / name)
    emb = np.eye(4, dtype="float32")
    emb[3] = emb[2] + 0.001                                      # b and c are identical in embedding space only
    df = pd.DataFrame({
        "image_path": [str(tmp_path / n) for n in ["a.png", "a_copy.png", "b.png", "c.png"]],
        "embeddings": [e.tolist() for e in emb],
        "timestamp": pd.date_range("2020-01-01", periods=4, tz="UTC"),
        "uploader": list("wxyz"), "categories": [["A"]] * 4,
    })
    res = cs.dedup_dataframe(df)
    assert res["dup_phash"].tolist() == [False, True, False, False]
    assert res["dup_embedding"].tolist() == [False, False, False, True]
    assert res["is_duplicate"].tolist() == [False, True, False, True]


def test_own_phash_is_stable_and_discriminative(tmp_path):
    rng = np.random.default_rng(4)
    base = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    Image.fromarray(base).save(tmp_path / "x.png")
    Image.fromarray(base).resize((256, 256)).save(tmp_path / "x_big.png")
    Image.fromarray(rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)).save(tmp_path / "y.png")
    hx, hb, hy = (cs.compute_phash(str(tmp_path / f)) for f in ("x.png", "x_big.png", "y.png"))
    assert cs.hamming_hex(hx, hb) <= 8
    assert cs.hamming_hex(hx, hy) > 12
