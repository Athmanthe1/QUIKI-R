"""CI smoke test: clean -> extract -> (train) -> load -> suggest -> fit stacker -> evaluate, on tiny synthetic data.
Uses FakeEncoder (colour-sensitive, deterministic) so no model download is needed. The torch training step is skipped
when torch is not installed."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import evaluate as ev
import extract_embeddings as ex
from cleaning_and_split import run_pipeline
from encoders import FakeEncoder
from engine import CategoryEngine
from fit_stacker import fit_from_rows
from manifest import read_manifest
from scoring import load_knowledge

COLS = {"Reds": (230, 20, 20), "Greens": (20, 200, 20), "Blues": (20, 20, 230)}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    rng = np.random.default_rng(0)
    rows = []
    n = 0
    for cat, col in COLS.items():
        for i in range(20):
            arr = np.clip(np.array(col) + rng.integers(-60, 60, size=(48, 48, 3)), 0, 255).astype("uint8")
            p = tmp / f"{cat}_{i}.png"
            Image.fromarray(arr).save(p)
            rows.append({"image_path": str(p), "categories": [cat, "Maintenance_tracking_2026"] if i % 5 == 0 else [cat],
                         "timestamp": (pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=n)).isoformat(),
                         "uploader": f"u{n // 3}", "title": f"{cat} photo {i}", "description": "", "phash": f"{int(rng.integers(0, 2**62)):016x}"})
            n += 1
    raw = pd.DataFrame(rows)
    return tmp, raw


def test_full_pipeline(built):
    tmp, raw = built
    tr, va, te, _rep = run_pipeline(raw)
    assert len(tr) and len(te)
    assert set(tr["uploader"]).isdisjoint(set(te["uploader"]))
    art = tmp / "artifacts"
    frames = {k: v for k, v in {"train": tr, "val": va, "test": te}.items() if len(v)}
    summary = ex.build_artifacts(frames, art, FakeEncoder(), min_freq=2, batch_size=8, store_factory=ex.MemoryStore,
                                 backend="numpy", dedup_threshold=1.01)
    man = read_manifest(art)
    assert man.classes and "Maintenance_tracking_2026" not in man.classes
    kn = load_knowledge(art, encoder=FakeEncoder())
    assert kn.index.ntotal == summary["n_index"] > 0

    eng = CategoryEngine(encoder=FakeEncoder(), artifact_dir=str(art), load_default=False)
    assert eng.status["mode"] == "artifacts"
    img = Image.fromarray(np.full((48, 48, 3), COLS["Reds"], dtype="uint8"))
    out = eng.suggest(img, {}, top_k=3, session=eng.new_session())
    assert out["suggestions"] and out["neighbors"]
    assert out["suggestions"][0]["category"] == "Reds"


def test_fit_and_evaluate_on_rows(built):
    tmp, raw = built
    enc = FakeEncoder()
    # build tiny in-memory knowledge directly from the same pipeline
    tr, va, te, _rep = run_pipeline(raw)
    art = tmp / "artifacts2"
    ex.build_artifacts({k: v for k, v in {"train": tr, "val": va, "test": te}.items() if len(v)}, art, enc, min_freq=2, batch_size=8,
                       store_factory=ex.MemoryStore, backend="numpy", dedup_threshold=1.01)
    kn = load_knowledge(art, encoder=enc)
    rng = np.random.default_rng(1)

    def rows_for(cat, n):
        out = []
        for _ in range(n):
            arr = np.clip(np.array(COLS[cat]) + rng.integers(-60, 60, size=(48, 48, 3)), 0, 255).astype("uint8")
            out.append({"path": f"{cat}", "categories": [cat], "width": 48, "height": 48,
                        "embedding": enc.encode_images([Image.fromarray(arr)])[0], "meta_embedding": None})
        return out
    val = rows_for("Reds", 6) + rows_for("Greens", 6) + rows_for("Blues", 6)
    kn.stacker = fit_from_rows(val, kn)
    test = rows_for("Reds", 5) + rows_for("Greens", 5) + rows_for("Blues", 5)
    man = read_manifest(art)
    counts = dict(zip(man.classes, man.class_counts))
    rep = ev.evaluate_rows(test, kn, counts, review_sample=5)
    assert rep["n_images_evaluated"] == 15
    assert rep["sample_wise"][1]["P"] > 0.8                      # colour classes are separable by construction
    assert rep["stacker"] == "fitted"
    assert rep["latency_scoring_only"]["p50_ms"] > 0
    assert "tail" in rep["frequency_buckets"] and "low (<256px)" in rep["resolution_band"]
    assert isinstance(rep["category_type"], str)                  # honest: not computed without a CSV
    assert "# QUIKI-R evaluation report" in ev.to_markdown(rep)
    json.dumps(rep, default=str)
