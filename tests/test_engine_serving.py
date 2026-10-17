"""Bugs 10-14 (+ wiring of 8, 9): read-only shared engine, neighbours, scorer health, honest state, no wasted startup."""
import json
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import extract_embeddings as ex
import scoring
from dag import save_category_graph
from encoders import FakeEncoder
from engine import CategoryEngine

COLORS = {"Reds": (230, 20, 20), "Greens": (20, 200, 20), "Blues": (20, 20, 230)}


def img(color, seed=0, size=32):
    rng = np.random.default_rng(seed)
    arr = np.clip(np.array(color)[None, None, :] + rng.integers(-40, 40, size=(size, size, 3)), 0, 255)
    return Image.fromarray(arr.astype("uint8"))


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("art")
    rows = []
    for cat, col in COLORS.items():
        for i in range(8):
            p = root / f"{cat}_{i}.png"
            img(col, seed=hash((cat, i)) % 1000).save(p)
            rows.append((str(p), [cat, "Colours"], f"2020-01-{len(rows) + 1:02d}"))
    df = pd.DataFrame({"image_path": [r[0] for r in rows], "categories": [r[1] for r in rows],
                       "timestamp": [r[2] for r in rows], "uploader": [f"u{i}" for i in range(len(rows))]})
    g = nx.DiGraph([("Colours", "Reds"), ("Colours", "Greens"), ("Colours", "Blues")])
    save_category_graph(root / "graph_in.json", g)
    out = root / "artifacts"
    ex.build_artifacts({"train": df}, out, FakeEncoder(), min_freq=2, batch_size=5, store_factory=ex.MemoryStore,
                       backend="numpy", graph_path=str(root / "graph_in.json"), dedup_threshold=1.01)
    return out


@pytest.fixture()
def engine(artifacts, tmp_path):
    return CategoryEngine(encoder=FakeEncoder(), artifact_dir=artifacts, cache_dir=tmp_path / "c")


# ---------------- #13 honest state ----------------
def test_state_reports_real_artifacts(engine):
    s = engine.state_summary()
    assert s["mode"] == "artifacts" and s["n_vectors"] == 24 and s["n_classes"] == 4 and s["load_error"] is None
    assert s["index_type"] == "flat" and s["calibrated"] is False and s["has_head"] is False


def test_state_is_demo_when_no_artifacts_and_says_so(tmp_path):
    e = CategoryEngine(encoder=FakeEncoder(), cache_dir=tmp_path)
    assert e.state_summary()["mode"] == "demo" and e.state_summary()["n_vectors"] == 0


def test_failed_artifact_load_falls_back_to_demo_with_the_reason(tmp_path):
    bad = tmp_path / "bad"; bad.mkdir()
    (bad / "manifest.json").write_text("{not json")
    e = CategoryEngine(encoder=FakeEncoder(), artifact_dir=bad, cache_dir=tmp_path / "c")
    s = e.state_summary()
    assert s["mode"] == "demo" and "ManifestError" in s["load_error"]          # banner must not say LIVE


def test_failed_load_does_not_leave_half_replaced_state(artifacts, tmp_path):
    e = CategoryEngine(encoder=FakeEncoder(), cache_dir=tmp_path)            # demo KB
    before = e.knowledge
    with pytest.raises(Exception):
        e.load_production_artifacts(tmp_path / "nowhere")
    assert e.knowledge is before


# ---------------- #2 real graph replaces toy graph ----------------
def test_real_graph_is_loaded_not_the_toy_one(engine):
    assert set(engine.category_graph.edges) == {("Colours", "Reds"), ("Colours", "Greens"), ("Colours", "Blues")}


# ---------------- #11 neighbours ----------------
def test_suggest_returns_neighbours_with_thumbnail_label_similarity(engine):
    out = engine.suggest(img(COLORS["Reds"], 5), top_k=3, n_neighbors=4)
    assert set(out) >= {"suggestions", "neighbors", "health", "state"}
    nb = out["neighbors"]
    assert len(nb) == 4 and [n["similarity"] for n in nb] == sorted((n["similarity"] for n in nb), reverse=True)
    assert all(isinstance(n["image"], Image.Image) and n["image"].size == (64, 64) for n in nb)
    assert all("Reds" in n["label"] for n in nb[:3])
    assert out["suggestions"][0]["category"] in {"Reds", "Colours"}


# ---------------- #8 wiring: honest fusion state ----------------
def test_missing_scorers_are_flagged_and_confidence_is_uncalibrated_without_stacker(engine):
    out = engine.suggest(img(COLORS["Greens"], 1), top_k=5)
    top = out["suggestions"][0]
    assert top["calibrated"] is False and set(top["missing"]) == {"cls", "meta"}
    assert 0.0 <= top["confidence"] <= 1.0


def test_demo_has_no_dummy_zero_vector_index(tmp_path):
    e = CategoryEngine(encoder=FakeEncoder(), cache_dir=tmp_path)
    out = e.suggest(img((10, 10, 10)), top_k=3)
    assert e.knowledge.index is None and out["health"]["knn"]["status"] == "unavailable"
    assert "knn" in out["suggestions"][0]["missing"] and "knn" not in out["suggestions"][0]["scores"]


# ---------------- #10 read-only shared engine ----------------
def test_feedback_lives_in_the_session_and_the_shared_engine_is_untouched(engine, tmp_path):
    a, b = engine.new_session(), engine.new_session()
    kn_before = (id(engine.knowledge), engine.knowledge.index.ntotal, engine.knowledge.head)
    q = img(COLORS["Blues"], 9)
    assert engine.record_feedback(q, "Reds", True, session=a)
    assert engine.record_feedback(q, "Greens", False, session=a)
    assert (id(engine.knowledge), engine.knowledge.index.ntotal, engine.knowledge.head) == kn_before
    assert a.counts()["images"] == 1 and b.counts()["images"] == 0 and b.overlay.ntotal == 0
    with pytest.raises(ValueError):
        engine.record_feedback(q, "Reds", True)                             # no session -> refuse, never a shared buffer


def test_accept_is_visible_to_knn_immediately_but_only_in_that_session(engine):
    a, b = engine.new_session(), engine.new_session()
    q = img((120, 120, 120), 3)                                              # grey: unlike anything indexed
    engine.record_feedback(q, "Greens", True, session=a)
    in_a = engine.suggest(q, top_k=5, n_neighbors=3, session=a)
    in_b = engine.suggest(q, top_k=5, n_neighbors=3, session=b)
    assert in_a["neighbors"][0]["source"] == "session" and in_a["neighbors"][0]["similarity"] > 0.999
    assert all(n["source"] == "index" for n in in_b["neighbors"])
    assert a.overlay.ntotal == 1


def test_reject_overrides_a_previous_accept_in_the_overlay(engine):
    s = engine.new_session()
    q = img((120, 120, 120), 3)
    engine.record_feedback(q, "Greens", True, session=s)
    engine.record_feedback(q, "Greens", False, session=s)
    out = engine.suggest(q, top_k=5, n_neighbors=3, session=s)
    assert all(n["source"] == "index" for n in out["neighbors"])             # emptied row must not vote
    assert next(iter(s.feedback.records.values())).negatives == {"Greens"}


def test_feedback_is_persisted_as_append_only_jsonl(engine, tmp_path):
    log = tmp_path / "fb" / "s.jsonl"
    s = engine.new_session(log_path=log)
    q = img(COLORS["Reds"], 2)
    engine.record_feedback(q, "Reds", True, session=s)
    engine.record_feedback(q, "Blues", False, session=s)
    lines = log.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["accepted"] is False


def test_unknown_category_feedback_is_ignored(engine):
    s = engine.new_session()
    assert engine.record_feedback(img(COLORS["Reds"]), "Not a class", True, session=s) is False
    assert s.counts()["images"] == 0


def test_training_needs_a_session_and_returns_zeros_when_empty(engine):
    with pytest.raises(ValueError):
        engine.train_supervised_head()
    assert engine.train_supervised_head(session=engine.new_session())["samples"] == 0


def test_training_runs_with_real_batches_and_session_head(engine):
    pytest.importorskip("torch")
    s = engine.new_session()
    for i in range(10):
        q = img(COLORS["Reds"], 100 + i)
        engine.record_feedback(q, "Reds", True, session=s)
        engine.record_feedback(q, "Blues", False, session=s)
    m = engine.train_supervised_head(epochs=3, lr=1e-2, batch_size=4, session=s)
    assert len(m["loss_history"]) == 3 and m["batches_per_epoch"] >= 2 and m["replay_rows"] > 0
    assert m["f1_held_out"] is True and s.head_trained and engine.knowledge.head is None   # shared engine still untouched


# ---------------- #12 scorer health ----------------
def test_scorer_exception_is_logged_and_reported_not_swallowed(engine, caplog, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("prototype matrix corrupted")
    monkeypatch.setattr(scoring, "siglip_text_scores", boom)
    with caplog.at_level(logging.WARNING, logger="CommonsScoring"):
        out = engine.suggest(img(COLORS["Reds"], 4), metadata={"title": "red thing"}, top_k=3)
    assert out["health"]["txt"]["status"] == "error" and "corrupted" in out["health"]["txt"]["detail"]
    assert out["health"]["meta"]["status"] == "error"
    assert out["health"]["knn"]["status"] == "ok" and out["suggestions"]            # other scorers still serve
    assert any("text scorer failed" in r.message for r in caplog.records)


def test_health_distinguishes_unavailable_from_ok(engine):
    h = engine.suggest(img(COLORS["Reds"], 4), top_k=3)["health"]
    assert h["knn"]["status"] == "ok" and h["txt"]["status"] == "ok"
    assert h["cls"]["status"] == "unavailable" and h["meta"]["status"] == "unavailable"


def test_text_encoder_truncates_long_descriptions():
    src = (Path(__file__).resolve().parent.parent / "encoders.py").read_text()
    assert "truncation=True" in src            # source-level: the real SigLIP path cannot run here


# ---------------- #14 no wasted startup work ----------------
class CountingEncoder(FakeEncoder):
    def __init__(self):
        super().__init__(); self.text_calls = 0
    def encode_texts(self, texts):
        self.text_calls += 1
        return super().encode_texts(texts)


def test_startup_uses_cached_prototypes_and_skips_the_demo_kb(artifacts, tmp_path):
    enc = CountingEncoder()
    e = CategoryEngine(encoder=enc, artifact_dir=artifacts, cache_dir=tmp_path / "c")
    assert e.state_summary()["mode"] == "artifacts"
    assert enc.text_calls == 0                                                # prototypes came from the cache on disk
    assert not (tmp_path / "c" / "demo").exists()                             # 50-category demo KB was never built


def test_demo_prototypes_are_cached_across_restarts(tmp_path):
    e1 = CategoryEngine(encoder=CountingEncoder(), cache_dir=tmp_path)
    enc2 = CountingEncoder()
    CategoryEngine(encoder=enc2, cache_dir=tmp_path)
    assert enc2.text_calls == 0


# ---------------- #11/#2 DAG through the engine ----------------
def test_dag_keeps_confident_parent_when_children_are_weak(engine):
    out = engine.suggest(img(COLORS["Reds"], 6), top_k=10, dag_threshold=0.99)   # nothing is 'selected' -> nothing suppressed
    cats = [s["category"] for s in out["suggestions"]]
    assert "Colours" in cats and "Reds" in cats
