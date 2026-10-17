"""Runs app.main() against the REAL engine (fake encoder) with a Streamlit stub: catches wiring errors between
the UI and the engine (state banner, session feedback, neighbours, scorer health). Not a UI/visual test."""
import importlib
import io
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import extract_embeddings as ex
from encoders import FakeEncoder
from engine import CategoryEngine
from tests.streamlit_stub import FakeUpload, install


def png_bytes(color, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.clip(np.array(color)[None, None, :] + rng.integers(-40, 40, size=(32, 32, 3)), 0, 255).astype("uint8")
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("app_art")
    rows = []
    for cat, col in {"Reds": (230, 20, 20), "Greens": (20, 200, 20)}.items():
        for i in range(6):
            p = root / f"{cat}{i}.png"; p.write_bytes(png_bytes(col, i)); rows.append((str(p), [cat], f"2020-01-{len(rows)+1:02d}"))
    df = pd.DataFrame({"image_path": [r[0] for r in rows], "categories": [r[1] for r in rows],
                       "timestamp": [r[2] for r in rows], "uploader": [f"u{i}" for i in range(len(rows))]})
    out = root / "art"
    ex.build_artifacts({"train": df}, out, FakeEncoder(), min_freq=2, batch_size=4, store_factory=ex.MemoryStore,
                       backend="numpy", dedup_threshold=1.01)
    return out


def load_app(upload):
    st = install(upload)
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    return st, app


def make_backend(app, engine):
    return app.Backend(engine.suggest, "live", engine.status["detail"], impl=engine)


def run_suggest(app, st, backend):
    app.load_backend = lambda: backend
    st.state["clicks"] = {"run_prediction"}
    st.session_state["run_requested"] = True       # what the Suggest button's on_click callback would set
    app.main()


def texts(st, kind):
    return [t for k, t in st.calls if k == kind]


def test_banner_and_results_with_real_artifacts(artifacts, tmp_path):
    st, app = load_app(FakeUpload(png_bytes((230, 20, 20), 99)))
    engine = CategoryEngine(encoder=FakeEncoder(), artifact_dir=artifacts, cache_dir=tmp_path)
    run_suggest(app, st, make_backend(app, engine))
    assert any("REAL ARTIFACTS" in t for t in texts(st, "success")) and any("UNCALIBRATED" in t for t in texts(st, "success"))
    pred = st.session_state["prediction"]
    assert pred["suggestions"] and pred["neighbors"] and pred["mode"] == "artifacts"
    assert all(n["label"] for n in pred["neighbors"]) and pred["health"]["knn"]["status"] == "ok"
    assert pred["suggestions"][0]["calibrated"] is False and "cls" in pred["suggestions"][0]["missing"]
    assert any(t.startswith("Scorer health") for k, t in st.calls if k == "expander")


def test_banner_is_demo_and_shows_load_error_when_artifacts_are_broken(tmp_path):
    bad = tmp_path / "bad"; bad.mkdir(); (bad / "manifest.json").write_text("{nope")
    st, app = load_app(FakeUpload(png_bytes((10, 10, 10))))
    engine = CategoryEngine(encoder=FakeEncoder(), artifact_dir=bad, cache_dir=tmp_path / "c")
    run_suggest(app, st, make_backend(app, engine))
    assert any("DEMO KNOWLEDGE BASE" in t for t in texts(st, "warning"))
    assert not texts(st, "success") or not any("LIVE" in t for t in texts(st, "success"))
    assert any("failed to load" in t and "ManifestError" in t for t in texts(st, "error"))
    assert st.session_state["prediction"]["neighbors"] == []
    assert any("No neighbours to show" in t and "no index" in t for t in texts(st, "info"))     # no canned 'latency' excuse


def test_mock_mode_banner(tmp_path):
    st, app = load_app(FakeUpload(png_bytes((10, 10, 10))))
    run_suggest(app, st, app.Backend(None, "mock", "engine.py unavailable: ImportError"))
    assert any("MOCK MODE" in t for t in texts(st, "warning"))
    assert st.session_state["prediction"]["mode"] == "mock"


def test_feedback_goes_to_the_visitor_session_not_the_shared_engine(artifacts, tmp_path):
    st, app = load_app(FakeUpload(png_bytes((230, 20, 20), 5)))
    engine = CategoryEngine(encoder=FakeEncoder(), artifact_dir=artifacts, cache_dir=tmp_path)
    backend = make_backend(app, engine)
    run_suggest(app, st, backend)
    img = st.session_state["_img"]
    app.record_feedback(img, "Reds", "accept")
    app.record_feedback(img, "Greens", "reject")
    sess = st.session_state["engine_session"]
    assert sess.counts() == {"images": 1, "positives": 1, "negatives": 1}
    assert engine.knowledge.head is None and engine.knowledge.index.ntotal == 12
    # a second visitor (fresh session_state) sees nothing of it
    st2, app2 = load_app(FakeUpload(png_bytes((230, 20, 20), 5)))
    app2.load_backend = lambda: make_backend(app2, engine)
    assert app2._session_counts(make_backend(app2, engine)) == {"images": 0, "positives": 0, "negatives": 0}


@pytest.mark.parametrize("mode", ["② Review & Feedback", "③ Train"])
def test_review_and_train_pages_render(artifacts, tmp_path, mode):
    st, app = load_app(FakeUpload(png_bytes((230, 20, 20), 7)))
    engine = CategoryEngine(encoder=FakeEncoder(), artifact_dir=artifacts, cache_dir=tmp_path)
    backend = make_backend(app, engine)
    run_suggest(app, st, backend)
    st.state["clicks"] = set(); st.session_state["mode"] = mode
    app.main()
    assert any("REAL ARTIFACTS" in t for t in texts(st, "success"))
    if mode.startswith("③"):
        assert any(t.startswith("Images with feedback") or "Images with feedback" in t for t in texts(st, "metric"))
