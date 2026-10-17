"""Wikimedia Commons automatic category suggestion dashboard.

Launch locally with:
    streamlit run app.py

Layout: the left sidebar is the control dashboard (mode selector, input
gateway, status). The main area shows the selected mode:

    Mode 1  Suggest             upload an image, get ranked categories + explainability
    Mode 2  Review & Feedback   accept / reject suggestions (human-in-the-loop), export
    Mode 3  Train               micro-train the supervised head on that feedback

The UI keeps the engine boundary forgiving: it can consume a module-level
prediction function, an engine object, or run in a clearly marked mock mode
while the model is unavailable.
"""

from __future__ import annotations

import importlib
import io
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Commons · Category Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


SUPPORTED_TYPES = ["jpg", "jpeg", "png"]
# Keys that can sit next to the category map in an engine response and must
# never be mistaken for category names.
NON_CATEGORY_KEYS = {"neighbors", "nearest_neighbors", "references", "knn", "metadata", "meta", "latency", "status", "health", "state"}
DEFAULT_SUGGESTIONS = [
    {
        "category": "Cultural heritage monuments",
        "confidence": 0.93,
        "path": ["Places", "Cultural heritage", "Cultural heritage monuments"],
        "reason": "Strong visual and semantic alignment with built heritage imagery.",
    },
    {
        "category": "Historic buildings",
        "confidence": 0.86,
        "path": ["Buildings", "Historic buildings", "Historic buildings"],
        "reason": "Architecture cues and title metadata reinforce this category.",
    },
    {
        "category": "Urban landscapes",
        "confidence": 0.74,
        "path": ["Urban studies", "Urban landscapes"],
        "reason": "Scene composition suggests a wider city context.",
    },
]

MODE_SUGGEST = "① Suggest"
MODE_REVIEW = "② Review & Feedback"
MODE_TRAIN = "③ Train"
MODES = [MODE_SUGGEST, MODE_REVIEW, MODE_TRAIN]
MODE_CAPTIONS = [
    "Upload an image and get ranked categories.",
    "Accept or reject suggestions to teach the engine.",
    "Retrain the classifier on your feedback.",
]
# (banner style, icon, title, what this mode does). Each mode gets its own colour so they
# stay easy to tell apart on the same page: blue / green / amber.
MODE_BANNERS = {
    MODE_SUGGEST: (
        "info", "🔎", "Mode 1 · Suggest",
        "Add an image (plus optional title, description and geotag) in the left panel, then press "
        "“Suggest categories”. Results here are read-only; give feedback in Mode 2.",
    ),
    MODE_REVIEW: (
        "success", "✅", "Mode 2 · Review & Feedback",
        "Human-in-the-loop curation. Accept correct categories (positive examples) and reject wrong ones "
        "(hard negatives). Every click is stored in your session straight away (accepts also join this session's nearest-neighbour index).",
    ),
    MODE_TRAIN: (
        "warning", "🧠", "Mode 3 · Train",
        "Teach the engine from your feedback: train a lightweight classifier head on the accepts and "
        "rejects you collected in Mode 2.",
    ),
}

GUIDE_MARKDOWN = """
**Modes**
- **① Suggest**: runs the engine on your image and ranks likely Commons categories.
- **② Review & Feedback**: you confirm (Accept) or correct (Reject) each suggestion; the engine learns from every click.
- **③ Train**: fits a small classifier head on your feedback (your session only; other visitors are not affected).

**Input gateway**
- **Image**: JPG or PNG. Sideways phone photos are auto-rotated; corrupt files are rejected.
- **Title / Description**: text clues about the picture. The engine compares them with every category name (the META scorer).
- **Geotag**: `latitude, longitude`. Shown as a map preview only; the current engine does not use it for ranking.
- **Suggestions to return**: how many categories to show, best first.

**Reading a result**
- **Confidence**: the estimated probability that the category is right (High ≥ 80%, Medium 50–79%, Low < 50%). It is calibrated by logistic stacking when the artifacts include `stacker.json`; otherwise it is an *uncalibrated* mean of the available scorers and the card says so.
- **KNN / TXT / META / CLS**: the individual scorers: vote of the nearest indexed images, image-to-category-name match (SigLIP), title/description match, and the trained classifier. A scorer that is not available (no index, no description, no trained head) is listed as missing and carries no weight.
- **Nearest neighbours**: the real indexed images behind the kNN vote (and images you accepted in this session).
- **📂 path**: where the category sits in the Commons category hierarchy.

**Training terms**
- **Accept = positive, Reject = hard negative**: a rejection is a stronger signal than a missing label.
- **Epochs / Learning rate / Batch size**: see the notes under each control in Mode 3. Batch size sets the real mini-batch size.
- **Asymmetric loss**: a loss that down-weights easy negatives and tolerates missing labels; a reject is a full-strength negative.
- **F1**: measured on held-out feedback images when there are enough of them, otherwise on the training feedback (the result says which).
"""


@dataclass
class Backend:
    """Resolved backend callable and its human-readable status. The engine inside is READ-ONLY and shared;
    per-visitor state lives in the Session returned by ``_session``."""

    predictor: Optional[Callable[..., Any]]
    status: str          # "live" (an engine answered) or "mock" (no engine; sample results)
    detail: str
    impl: Any = None     # the shared CategoryEngine (None in mock mode)

    @property
    def state(self) -> dict[str, Any]:
        if self.impl is not None and hasattr(self.impl, "state_summary"):
            return self.impl.state_summary()
        return {"mode": "mock", "detail": self.detail, "load_error": None, "n_classes": 0, "n_vectors": 0,
                "index_type": "none", "calibrated": False, "has_head": False, "notes": []}

    @property
    def mode(self) -> str:
        """'artifacts' (real data), 'demo' (toy knowledge base) or 'mock' (no engine)."""
        return "mock" if self.impl is None else str(self.state.get("mode", "demo"))

    @property
    def can_learn(self) -> bool:
        return self.status == "live" and callable(getattr(self.impl, "record_feedback", None))

    @property
    def can_train(self) -> bool:
        return self.status == "live" and callable(getattr(self.impl, "train_supervised_head", None))

    def send_feedback(self, image: Image.Image, category: str, verdict: str, session: Any) -> bool:
        """Forward an accept/reject to THIS visitor's session; rejects become hard negatives."""
        if not self.can_learn or image is None or session is None:
            return False
        return bool(self.impl.record_feedback(image, category, accepted=(verdict == "accept"), session=session))


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _confidence(value: Any) -> float:
    try:
        number = float(value)
        if number > 1:
            number /= 100
        return max(0.0, min(1.0, number))
    except (TypeError, ValueError):
        return 0.0


def _first(mapping: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _normalise_path(value: Any, category: str) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split("/") if part.strip()] or [category]
    if isinstance(value, (list, tuple)):
        return [_safe_text(part) for part in value if _safe_text(part)] or [category]
    return [category]


def _normalise_suggestions(raw: Any) -> list[dict[str, Any]]:
    """Convert common engine response shapes into a stable display contract."""
    if isinstance(raw, Mapping):
        raw_items = _first(raw, ["suggestions", "predictions", "categories", "results"], None)
        if raw_items is None:
            # The response itself may be a {category: score} map; drop sibling metadata keys.
            raw_items = {key: value for key, value in raw.items() if key not in NON_CATEGORY_KEYS}
    else:
        raw_items = raw

    if isinstance(raw_items, Mapping):
        # Values may be plain scores ({"Cat": 0.9}) rather than dicts.
        raw_items = [
            dict(value, category=key) if isinstance(value, Mapping) else {"category": key, "confidence": value}
            for key, value in raw_items.items()
        ]
    if not isinstance(raw_items, (list, tuple)):
        raw_items = []

    suggestions: list[dict[str, Any]] = []
    for item in raw_items:
        scorer_map: Optional[Mapping[str, Any]] = None
        if isinstance(item, str):
            category, score, path, reason = item, 0.0, [item], "Returned by the category engine."
        elif isinstance(item, Mapping):
            category = _safe_text(_first(item, ["category", "label", "name", "title"], "Unlabelled category"))
            # "score" may be a per-scorer dict in the V1.2 engine; only scalars count as confidence.
            raw_score = _first(item, ["confidence", "probability", "similarity", "score"], 0)
            score = _confidence(raw_score if not isinstance(raw_score, Mapping) else 0)
            path = _normalise_path(_first(item, ["path", "hierarchy", "dag_path", "parents"]), category)
            reason = _safe_text(_first(item, ["reason", "explanation", "rationale"], "Suggested by multimodal evidence."))
            scorer_map = item.get("scores") if isinstance(item.get("scores"), Mapping) else None
            if scorer_map is None and isinstance(item.get("score"), Mapping):
                scorer_map = item["score"]
        elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
            # (category, score) pairs
            category, score = item[0], _confidence(item[1])
            path, reason = [category], "Returned by the category engine."
        else:
            category, score, path, reason = _safe_text(item), 0.0, [_safe_text(item)], "Returned by the category engine."
        if category:
            entry = {"category": category, "confidence": score, "path": path, "reason": reason}
            if scorer_map:
                entry["scores"] = {str(k): _confidence(v) for k, v in scorer_map.items()}
            if isinstance(item, Mapping):
                entry["calibrated"] = item.get("calibrated")
                entry["missing"] = [str(m) for m in (item.get("missing") or [])]
            suggestions.append(entry)
    return suggestions


def _normalise_neighbors(raw: Any) -> list[dict[str, Any]]:
    """Read optional nearest-neighbor data without making the engine contract rigid."""
    if isinstance(raw, Mapping):
        raw = _first(raw, ["neighbors", "nearest_neighbors", "references", "knn"], [])
    if not isinstance(raw, (list, tuple)):
        return []
    result = []
    for index, item in enumerate(raw[:6], start=1):
        if isinstance(item, Mapping):
            image = _first(item, ["image", "image_path", "path", "url", "thumbnail", "data"])
            label = _safe_text(_first(item, ["category", "label", "title", "name"], f"Reference {index}"))
            score = _confidence(_first(item, ["similarity", "score", "confidence"], 0))
        else:
            image, label, score = item, f"Reference {index}", 0.0
        result.append({"image": image, "label": label, "score": score})
    return result


def _call_predictor(predictor: Callable[..., Any], image: Image.Image, metadata: dict[str, str], top_k: int,
                    session: Any = None) -> Any:
    payload = {
        "image": image,
        "image_input": image,  # V1.2 engine: suggest(image_input, metadata=None, top_k=10)
        "title": metadata["title"],
        "description": metadata["description"],
        "geotag": metadata["geotag"],
        "metadata": metadata,
        "top_k": top_k,
        "session": session,
    }
    try:
        signature = inspect.signature(predictor)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        if accepts_kwargs:
            return predictor(**payload)
        accepted = {key: value for key, value in payload.items() if key in signature.parameters}
        if accepted:
            return predictor(**accepted)
        return predictor(image)
    except (TypeError, ValueError):
        try:
            return predictor(image=image, title=metadata["title"], description=metadata["description"], geotag=metadata["geotag"])
        except TypeError:
            return predictor(image)  # last resort for engines taking only the image as first positional arg


def _find_artifact_dir() -> Optional[str]:
    """Folder holding manifest.json: $QUIKI_ARTIFACTS_DIR, ./artifacts, or <app dir>/artifacts. If none exists and
    $QUIKI_HF_REPO is set, try to download the artifacts from the Hugging Face Hub (needs network)."""
    candidates = [os.environ.get("QUIKI_ARTIFACTS_DIR"), "artifacts", str(Path(__file__).resolve().parent / "artifacts")]
    for c in candidates:
        if c and (Path(c) / "manifest.json").is_file():
            return str(c)
    repo = os.environ.get("QUIKI_HF_REPO")
    if repo:
        try:
            from huggingface_hub import snapshot_download
            target = Path(__file__).resolve().parent / "artifacts"
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(target))
            if (target / "manifest.json").is_file():
                return str(target)
        except Exception as exc:
            logger.warning("could not download artifacts from %s: %s", repo, exc)
    return None


@st.cache_resource(show_spinner=False)
def load_backend() -> Backend:
    """Build the shared, read-only engine once. Failures stay visible (mock mode) but never break the UI."""
    try:
        module = importlib.import_module("engine")
    except Exception as exc:  # backend libraries can fail during import
        return Backend(None, "mock", f"engine.py unavailable: {type(exc).__name__}: {exc}")
    try:
        engine = module.CategoryEngine(artifact_dir=_find_artifact_dir())
    except Exception as exc:
        return Backend(None, "mock", f"engine initialisation failed: {type(exc).__name__}: {exc}")
    return Backend(engine.suggest, "live", str(engine.status.get("detail", "")), impl=engine)


def _session(backend: Backend) -> Any:
    """This visitor's engine Session (feedback, overlay index, fine-tuned head), kept in st.session_state."""
    if not backend.can_learn:
        return None
    session = st.session_state.get("engine_session")
    if session is None:
        session = backend.impl.new_session()
        session.log_path = Path("feedback") / f"{session.id}.jsonl"   # append-only JSONL, one file per session
        st.session_state["engine_session"] = session
    return session


def _mock_result(image: Image.Image, metadata: dict[str, str], top_k: int) -> dict[str, Any]:
    """Deterministic demo output so judges can explore every UI state offline."""
    title = metadata["title"] or "untitled upload"
    suggestions = [dict(item) for item in DEFAULT_SUGGESTIONS[:top_k]]
    suggestions[0]["reason"] = f'Visual match plus metadata context from “{title}”.'
    return {"suggestions": suggestions, "neighbors": []}


def _dag_path(graph: Any, name: str) -> list[str]:
    """Root-to-category path through the engine's category DAG (first parent, alphabetically, at each step)."""
    path, seen = [name], {name}
    while True:
        parents = sorted(p for p in graph.predecessors(path[0]) if p not in seen)
        if not parents:
            return path
        path.insert(0, parents[0])
        seen.add(parents[0])


def _attach_dag_paths(suggestions: list[dict[str, Any]], impl: Any) -> None:
    """The V1.2 engine does not return hierarchy paths, so read them from its category graph."""
    graph = getattr(impl, "category_graph", None)
    if graph is None or not hasattr(graph, "predecessors"):
        return
    for item in suggestions:
        try:
            if item.get("path") in (None, [item["category"]]) and item["category"] in graph:
                item["path"] = _dag_path(graph, item["category"])
        except Exception:  # a missing or odd graph must never break the results
            continue


def _run_prediction(backend: Backend, image: Image.Image, metadata: dict[str, str], top_k: int,
                    session: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    if backend.predictor is None:
        result = _mock_result(image, metadata, top_k)
    else:
        raw = _call_predictor(backend.predictor, image, metadata, top_k, session)
        neighbors = _normalise_neighbors(raw) if isinstance(raw, Mapping) else []
        result = {"suggestions": _normalise_suggestions(raw), "neighbors": neighbors}
        if isinstance(raw, Mapping):
            result["health"] = raw.get("health") or {}
            result["state"] = raw.get("state") or {}
        _attach_dag_paths(result["suggestions"], backend.impl)
    # Rank best-first and respect the slider even if the engine returns more.
    result["suggestions"] = sorted(result["suggestions"], key=lambda s: s["confidence"], reverse=True)[:top_k]
    result["latency"] = time.perf_counter() - started
    result["mode"] = backend.mode
    return result


def _image_from_upload(uploaded_file: Any) -> Optional[Image.Image]:
    if uploaded_file is None:
        return None
    if uploaded_file.type not in {"image/jpeg", "image/png"}:
        st.error("Unsupported file type. Please upload a JPG or PNG image.")
        return None
    try:
        raw = uploaded_file.getvalue()
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        # exif_transpose fixes sideways phone photos before they reach the model.
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        return image
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This file could not be read as an image. Please choose a valid JPG or PNG.")
        return None


def _parse_geotag(value: str) -> Optional[tuple[float, float]]:
    """Return (lat, lon) for strings like '51.0504, 13.7373', else None."""
    parts = [part for part in value.replace(";", ",").replace(" ", ",").split(",") if part]
    if len(parts) != 2:
        return None
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


def _display_neighbor(image: Any, fallback: Optional[Image.Image]) -> Any:
    if image is None:
        return fallback
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(image))
        except (UnidentifiedImageError, OSError):
            return fallback
    return image


def _tier(score: float) -> tuple[str, str]:
    """Colour + label for a confidence value."""
    if score >= 0.8:
        return "green", "High"
    if score >= 0.5:
        return "orange", "Medium"
    return "red", "Low"


# ---------------------------------------------------------------------------
# Session state and feedback
# ---------------------------------------------------------------------------

def _verdicts() -> dict[str, str]:
    """Per-category human verdicts for the current result: {category: 'accept' | 'reject'}."""
    return st.session_state.setdefault("verdicts", {})


def _reset_review() -> None:
    st.session_state["verdicts"] = {}


def _session_counts(backend: Backend) -> dict[str, int]:
    """Feedback held by THIS visitor's session, after later clicks have overridden earlier ones."""
    session = st.session_state.get("engine_session")
    if session is None or not backend.can_learn:
        return {"images": 0, "positives": 0, "negatives": 0}
    return session.counts()


def record_feedback(image: Optional[Image.Image], category: str, verdict: str) -> None:
    """Button callback: remember the verdict locally and forward it to the live engine.

    The engine receives the PIL image itself (it extracts the embedding); only the single
    current query image is held in session state, so nothing accumulates between runs.
    """
    _verdicts()[category] = verdict
    if image is None:
        image = st.session_state.get("_img")  # the engine needs the query image to embed it
    backend = load_backend()
    if not backend.can_learn:
        st.toast("Saved locally. Continual learning needs the live engine.", icon="ℹ️")
        return
    try:
        if backend.send_feedback(image, category, verdict, _session(backend)):
            kind = "positive" if verdict == "accept" else "hard negative"
            st.toast(f"Recorded “{category}” as a {kind}.", icon="✅" if verdict == "accept" else "🚫")
        else:
            st.toast("No query image available, so feedback was not sent to the engine.", icon="⚠️")
    except Exception as exc:  # a feedback failure must never take the UI down
        st.toast(f"Engine rejected the feedback: {type(exc).__name__}", icon="⚠️")


def _on_verdict(category: str, verdict: str) -> None:
    record_feedback(st.session_state.get("_img"), category, verdict)


def _request_run() -> None:
    """Suggest-button callback: queue a prediction and jump to Mode 1 to show it."""
    st.session_state["run_requested"] = True
    st.session_state["mode"] = MODE_SUGGEST


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_header(backend: Backend) -> None:
    st.title("Commons Category Intelligence")
    st.caption("Multimodal suggestions for a cleaner, more discoverable Wikimedia Commons")
    state = backend.state
    mode = backend.mode
    if mode == "artifacts":
        calibration = "calibrated probabilities" if state.get("calibrated") else "UNCALIBRATED scores (no stacker.json)"
        st.success(f"● REAL ARTIFACTS  ·  {state.get('detail')}  ·  {calibration}")
        for note in state.get("notes", []):
            st.caption(f"ℹ️ {note}")
    elif mode == "demo":
        st.warning("● DEMO KNOWLEDGE BASE  ·  50 built-in toy categories, no image index, no trained head. "
                   "Suggestions are NOT based on Commons data.")
        if state.get("load_error"):
            st.error(f"Production artifacts were found but failed to load: {state['load_error']}")
    else:
        st.warning(f"● MOCK MODE  ·  {backend.detail}. The dashboard stays navigable with fixed sample results.")


def _render_mode_banner(mode: str) -> None:
    style, icon, title, text = MODE_BANNERS[mode]
    getattr(st, style)(f"{icon} **{title}**  ·  {text}")


def _render_input() -> tuple[Optional[Image.Image], dict[str, str], int]:
    """Input gateway, pinned in the sidebar so it stays available in every mode."""
    with st.sidebar.expander("📥 Input gateway", expanded=True):
        uploaded = st.file_uploader(
            "Image", type=SUPPORTED_TYPES, key="image_upload",
            help="The Commons image to categorise. JPG or PNG; sideways phone photos are auto-rotated and corrupt files are rejected.",
        )
        image = _image_from_upload(uploaded)
        if image is not None:
            st.image(image, caption=f"{image.width} × {image.height}px · RGB", use_container_width=True)
        else:
            st.info("Upload a JPG or PNG to begin. Your metadata stays in this session.")

        title = st.text_input(
            "Image title", placeholder="e.g. South façade of the old town hall", key="metadata_title",
            help="A short caption. The engine compares it with each category name (the META scorer) to favour more specific categories. Optional.",
        )
        description = st.text_area(
            "Description", placeholder="Describe the visible subject, place, or event", height=90, key="metadata_description",
            help="Longer free text about what the image shows. It is combined with the title for the META scorer, so more detail gives it more to work with. Optional.",
        )
        geotag = st.text_input(
            "Geotag", placeholder="e.g. 51.0504, 13.7373", key="metadata_geotag",
            help="Where the photo was taken, as “latitude, longitude”. A valid value shows a map preview. The current engine does not use it for ranking. Optional.",
        )

        # Live geotag feedback: confirm valid coordinates on a map, flag typos early.
        geotag_clean = _safe_text(geotag)
        if geotag_clean:
            coords = _parse_geotag(geotag_clean)
            if coords:
                st.caption(f"📍 {coords[0]:.4f}, {coords[1]:.4f}")
                st.map(pd.DataFrame({"lat": [coords[0]], "lon": [coords[1]]}), zoom=11, height=140)
            else:
                st.caption("⚠️ Geotag not recognised. Use “latitude, longitude”, e.g. 51.0504, 13.7373. It is passed along as plain text and is not used for ranking.")

        top_k = st.slider(
            "Suggestions to return", min_value=3, max_value=8, value=3, key="top_k",
            help="How many categories to show, best first. Also trims the list if the engine returns more.",
        )
        st.button(
            "Suggest categories", type="primary", use_container_width=True, disabled=image is None,
            key="run_prediction", on_click=_request_run,
            help="Runs the engine on this image and metadata, then opens Mode 1 with the ranked results.",
        )
    metadata = {"title": _safe_text(title), "description": _safe_text(description), "geotag": geotag_clean}
    return image, metadata, top_k


def _render_suggestion_card(item: dict[str, Any], index: int, interactive: bool) -> None:
    score = _confidence(item.get("confidence"))
    colour, label = _tier(score)
    category = item["category"]
    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"**{index:02d}  {category}**  :{colour}[● {label}]")
            st.caption(item.get("reason", "Multimodal evidence"))
        with right:
            calibrated = item.get("calibrated")
            label = "Probability" if calibrated else ("Score (uncalibrated)" if calibrated is False else "Confidence")
            st.metric(label, f"{score:.0%}", help="Estimated probability that this category is right. Calibrated by logistic "
                      "stacking when stacker.json is present; otherwise an uncalibrated mean of the available scorers.")
        st.progress(score, text=f"{('Probability' if item.get('calibrated') else 'Score')} · {score:.0%}")
        st.caption("📂  " + "  ›  ".join(item.get("path", [category])))

        scores = item.get("scores")
        if scores:
            cols = st.columns(len(scores))
            for col, (name, value) in zip(cols, scores.items()):
                col.caption(f"{name.upper()} · {value:.0%}")
                col.progress(value)
        if item.get("missing"):
            st.caption("Not available for this image: " + ", ".join(m.upper() for m in item["missing"]) +
                       " (no weight given; see Scorer health)")

        verdict = _verdicts().get(category)
        if not interactive:
            if verdict == "accept":
                st.caption(":green[✔ Accepted in Mode 2]")
            elif verdict == "reject":
                st.caption(":red[✘ Rejected in Mode 2]")
            return
        accept_col, reject_col, status_col = st.columns([1, 1, 1.4])
        accept_col.button(
            "✓ Accept", key=f"accept_btn_{index}_{category}", use_container_width=True,
            type="primary" if verdict == "accept" else "secondary",
            on_click=_on_verdict, args=(category, "accept"),
            help="This category is correct. Sent to the engine as a positive example.",
        )
        reject_col.button(
            "✗ Reject", key=f"reject_btn_{index}_{category}", use_container_width=True,
            type="primary" if verdict == "reject" else "secondary",
            on_click=_on_verdict, args=(category, "reject"),
            help="This category is wrong. Sent to the engine as a hard negative, a stronger signal than a missing label.",
        )
        if verdict == "accept":
            status_col.markdown(":green[✔ Accepted]")
        elif verdict == "reject":
            status_col.markdown(":red[✘ Rejected · hard negative]")


def _render_suggestions(result: Optional[dict[str, Any]], interactive: bool) -> bool:
    """Ranked category cards. Returns True when there is something to show."""
    with st.container(border=True):
        st.subheader("Suggested categories" if not interactive else "Review suggestions")
        if not result:
            st.info("Your ranked categories will appear here after you run the suggestion engine.")
            return False
        suggestions = result.get("suggestions", [])
        if not suggestions:
            st.warning("The engine returned no categories for this image. Try adding a title or description.")
            return False
        if not interactive:
            best = suggestions[0]
            metric_a, metric_b, metric_c, metric_d = st.columns(4)
            metric_a.metric("Top match", best["category"], help="The highest-ranked category.")
            metric_b.metric("Confidence", f"{_confidence(best['confidence']):.0%}", help="Confidence of the top match.")
            metric_c.metric("Labels returned", len(suggestions), help="How many categories came back (set by “Suggestions to return”).")
            metric_d.metric("Latency", f"{result.get('latency', 0):.2f}s", help="Time the engine took for this image.")
        else:
            st.caption("Accept what is right, reject what is wrong. Each click goes to the engine immediately; use Mode 3 to train on it.")
        for index, suggestion in enumerate(suggestions, start=1):
            _render_suggestion_card(suggestion, index, interactive)
        return True


def _render_evidence(image: Optional[Image.Image], result: dict[str, Any]) -> None:
    with st.container(border=True):
        st.subheader("Explainability · nearest-neighbor evidence")
        st.caption("Indexed images that voted for the categories above (64px thumbnails). Images you accepted in this "
                   "session are marked as such.")
        neighbors = result.get("neighbors") or []
        if not neighbors:
            reason = (result.get("health") or {}).get("knn", {}).get("detail")
            if result.get("mode") == "mock":
                st.info("Mock mode: no reference thumbnails are generated. The uploaded image is shown as the query anchor.")
                if image is not None:
                    st.image(image, caption="Uploaded image · query anchor", width=180)
            else:
                st.info(f"No neighbours to show: {reason or 'the kNN scorer returned nothing'}.")
            return
        columns = st.columns(min(3, len(neighbors)))
        for index, neighbor in enumerate(neighbors):
            with columns[index % len(columns)]:
                reference = _display_neighbor(neighbor.get("image"), None)
                if reference is not None:
                    try:
                        st.image(reference, use_container_width=True)
                    except Exception:  # broken thumbnail must not take the page down
                        st.caption("🖼️ Thumbnail unavailable")
                else:
                    st.caption("🖼️ No thumbnail stored for this item")
                source = " · this session" if neighbor.get("source") == "session" else ""
                st.caption(f"{neighbor['label']} · {_confidence(neighbor['score']):.0%} similarity{source}")


_HEALTH_ICON = {"ok": "✅", "unavailable": "⚪", "error": "❌"}


def _render_health(result: dict[str, Any]) -> None:
    """Scorer health: which scorers ran, which were unavailable and why, which failed."""
    health = result.get("health") or {}
    if not health:
        return
    failing = [n for n, h in health.items() if h.get("status") == "error"]
    with st.expander("Scorer health" + (f" · ⚠️ {len(failing)} failing" if failing else ""), expanded=bool(failing)):
        for name in ("knn", "txt", "meta", "cls"):
            if name in health:
                h = health[name]
                st.markdown(f"{_HEALTH_ICON.get(h.get('status'), '•')} **{name.upper()}** · {h.get('status')}"
                            + (f" · {h.get('detail')}" if h.get("detail") else ""))
        if failing:
            st.caption("A failing scorer is logged on the server and ignored for this result; the others still count.")


def _render_review_export(result: dict[str, Any]) -> None:
    """Review table plus Commons-ready export of accepted (or all non-rejected) categories."""
    suggestions = result.get("suggestions", [])
    if not suggestions:
        return
    verdicts = _verdicts()
    accepted = [item for item in suggestions if verdicts.get(item["category"]) == "accept"]
    rejected = [item for item in suggestions if verdicts.get(item["category"]) == "reject"]
    # With no explicit accepts, export everything the reviewer has not rejected.
    chosen = accepted or [item for item in suggestions if item not in rejected]
    with st.container(border=True):
        st.subheader("Review summary & export")
        st.caption(
            f"{len(accepted)} accepted · {len(rejected)} rejected of {len(suggestions)}."
            + ("" if accepted else " Nothing accepted yet, so the export includes all non-rejected suggestions.")
        )

        rows = [
            {"Rank": index, "Category": item["category"], "Confidence": f"{_confidence(item['confidence']):.1%}",
             "Path": " › ".join(item.get("path", [])), "Verdict": verdicts.get(item["category"], "—")}
            for index, item in enumerate(suggestions, start=1)
        ]
        with st.expander("Open compact review table"):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        wikitext = "\n".join(f"[[Category:{item['category']}]]" for item in chosen)
        st.markdown("**Paste into the Commons file page**")
        st.caption("Ready-made category tags for the file's wiki page.")
        st.code(wikitext, language="text")

        export_rows = [
            {"category": item["category"], "confidence": round(_confidence(item["confidence"]), 4), "path": item.get("path", [])}
            for item in chosen
        ]
        csv_bytes = pd.DataFrame(
            [{**row, "path": " > ".join(row["path"])} for row in export_rows]
        ).to_csv(index=False).encode("utf-8")
        left, right = st.columns(2)
        left.download_button("Download CSV", csv_bytes, file_name="commons_categories.csv", mime="text/csv",
                             use_container_width=True, help="The chosen categories as a spreadsheet-friendly file.")
        right.download_button(
            "Download JSON", json.dumps(export_rows, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="commons_categories.json", mime="application/json", use_container_width=True,
            help="The chosen categories with confidence and hierarchy path, for scripts and tools.",
        )


def _page_suggest(image: Optional[Image.Image], result: Optional[dict[str, Any]]) -> None:
    if _render_suggestions(result, interactive=False) and result:
        st.write("")
        _render_evidence(image if image is not None else st.session_state.get("_img"), result)
        _render_health(result)


def _page_review(result: Optional[dict[str, Any]]) -> None:
    if not result or not result.get("suggestions"):
        st.info("Nothing to review yet. Run **① Suggest** first (use “Suggest categories” in the left panel).")
        return
    left, right = st.columns([0.8, 1.6], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Query image")
            query_image = st.session_state.get("_img")
            if query_image is not None:
                st.image(query_image, use_container_width=True)
            meta = st.session_state.get("last_metadata", {})
            if meta.get("title"):
                st.caption(f"**Title:** {meta['title']}")
            if meta.get("geotag"):
                st.caption(f"**Geotag:** {meta['geotag']}")
            counts = _session_counts(load_backend())
            st.metric("Accepted (current)", counts["positives"], help="Positive examples in your session. Changing your mind on a category replaces the earlier click.")
            st.metric("Rejected (current)", counts["negatives"], help="Hard negatives in your session.")
    with right:
        _render_suggestions(result, interactive=True)
    st.write("")
    _render_review_export(result)


def _float_list(value: Any) -> list[float]:
    """Coerce an engine-provided sequence (list, tuple, array, Series) into plain floats."""
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return []
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return []


def _final_loss(metrics: Mapping[str, Any]) -> Optional[float]:
    """Final loss: the engine's 'final_loss' (older builds: 'loss'), else the last 'loss_history' point."""
    for key in ("final_loss", "loss"):
        try:
            if metrics.get(key) is not None:
                return float(metrics[key])
        except (TypeError, ValueError):
            continue
    history = _float_list(metrics.get("loss_history"))
    return history[-1] if history else None


def _f1_score(metrics: Mapping[str, Any]) -> Optional[float]:
    """F1 under any of the common key spellings; None when the engine did not report it."""
    for key in ("f1_score", "f1", "f1-score", "F1"):
        try:
            if metrics.get(key) is not None:
                return float(metrics[key])
        except (TypeError, ValueError):
            continue
    return None


def _page_train(backend: Backend) -> None:
    """Mode 3: micro-train the supervised head on accumulated accept/reject feedback."""
    st.caption(
        "The engine uses a frozen SigLIP encoder. You can train a lightweight Supervised Multi-Label Head "
        "using Asymmetric Loss on the feedback you've provided (Accepts = Positives, Rejects = Hard Negatives)."
    )
    live = backend.can_train
    if not live:
        st.info("Training requires the live engine.")

    counts = _session_counts(backend)
    with st.container(border=True):
        st.subheader("Feedback buffer")
        st.caption("Your session's feedback from Mode 2, stored per image (a later click on the same category replaces the earlier one).")
        a, b, c = st.columns(3)
        a.metric("Positives (accepts)", counts["positives"], help="Categories you confirmed as correct.")
        b.metric("Hard negatives (rejects)", counts["negatives"], help="Categories you marked as wrong.")
        c.metric("Images with feedback", counts["images"], help="Training rows: one per image, not one per click.")

    controls, outcome = st.columns([1, 1.2], gap="large")
    with controls:
        with st.container(border=True):
            st.subheader("Training controls")
            epochs = st.slider("Epochs", 1, 20, 5, key="train_epochs", disabled=not live,
                               help="Passes over your feedback (plus a fresh replay sample of indexed training images each pass).")
            st.caption("More epochs fit your feedback more tightly, but a small buffer can be over-fitted.")
            lr = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=1e-4, format="%.5f",
                                 key="train_lr", disabled=not live, help="Size of each update step.")
            st.caption("Too high makes the loss jump around; too low barely changes the classifier.")
            batch_size = st.slider("Batch Size", 8, 64, 32, key="train_batch", disabled=not live,
                                   help="Images per update step: real mini-batches.")
            st.caption("Smaller batches mean more, noisier updates per epoch; larger ones are steadier. The result shows batches per epoch.")
            run_training = st.button(
                "🚀 Run Micro-Training on Feedback Buffer", type="primary", key="run_training",
                disabled=not live, use_container_width=True,
                help="Trains the linear head on top of the frozen embeddings using the feedback buffer.",
            )

    if run_training:
        with st.spinner("Training the supervised head on the feedback buffer…"):
            try:
                metrics = backend.impl.train_supervised_head(epochs=epochs, lr=float(lr), batch_size=batch_size,
                                                             session=_session(backend))
            except Exception as exc:
                st.error(f"Training could not complete: {type(exc).__name__}: {exc}")
            else:
                metrics = dict(metrics or {})
                st.session_state["last_training"] = metrics
                if int(metrics.get("samples") or 0) > 0:
                    st.toast("Head updated for your session. The CLS scorer now uses it.", icon="🧠")
                else:
                    st.toast("Nothing to train on yet. Accept or reject suggestions in Mode 2 first.", icon="⚠️")

    with outcome:
        with st.container(border=True):
            st.subheader("Results")
            last = st.session_state.get("last_training")
            if last and not int(last.get("samples") or 0):
                # The engine returns zeros when its feedback buffer is empty; do not present that as a trained head.
                st.warning("Nothing was trained: the feedback buffer is empty. Accept or reject suggestions in "
                           "**② Review & Feedback**, then run training again.")
            elif last:
                st.success("Head updated for your session. The CLS scorer now uses it; other visitors are not affected.")
                f1_col, loss_col, samples_col = st.columns(3)
                f1 = _f1_score(last)
                held_out = bool(last.get("f1_held_out"))
                f1_col.metric(f"F1-Score ({'held-out' if held_out else 'in-sample'})", f"{f1:.3f}" if f1 is not None else "—",
                              help="Micro F1 over labelled entries only (your accepts and rejects). 'Held-out' = measured on images not "
                                   "used for fitting; 'in-sample' = too little feedback to hold any out, so it is optimistic.")
                final_loss = _final_loss(last)
                loss_col.metric("Final Loss", f"{final_loss:.4f}" if final_loss is not None else "—",
                                help="Asymmetric loss after the last epoch; lower is better.")
                samples_col.metric("Images", last.get("samples", 0), help="Images with feedback used in the last run.")
                if last.get("batches_per_epoch"):
                    st.caption(f"{last.get('batches_per_epoch')} mini-batches per epoch · {last.get('replay_rows', 0)} replay rows "
                               f"from the indexed training data · {last.get('held_out_images', 0)} images held out")

                loss_curve = _float_list(last.get("loss_history"))
                if loss_curve:
                    st.caption("Loss per epoch (it should drop as training progresses)")
                    st.line_chart(pd.DataFrame({"Asymmetric Loss": loss_curve}, index=range(1, len(loss_curve) + 1)))
                else:
                    st.caption("The engine did not return a per-epoch loss history for this run.")
            else:
                st.caption("Run a training pass to see F1, loss, sample count and the loss curve here.")


def main() -> None:
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "last_metadata" not in st.session_state:
        st.session_state.last_metadata = {}

    backend = load_backend()

    # ---- left dashboard: appearance, mode selector, input gateway, status ----
    with st.sidebar:
        st.subheader("◈ Commons Intelligence")
        st.radio(
            "Mode", MODES, key="mode", captions=MODE_CAPTIONS,
            help="Three stages on one page: get suggestions, review them, then train on your feedback. Your results carry over when you switch.",
        )
        st.divider()
    image, metadata, top_k = _render_input()
    with st.sidebar:
        st.divider()
        st.metric("Engine state", {"artifacts": "Real data", "demo": "Demo KB"}.get(backend.mode, "Mock"),
                  help="Real data = the engine loaded your artifacts. Demo KB = built-in toy categories, not Commons data. "
                       "Mock = no engine, fixed sample results.")
        if st.button("Clear current result", use_container_width=True, key="clear_prediction",
                     help="Discards the current results, your accept/reject marks and the held query image. Training history is kept."):
            st.session_state.prediction = None
            st.session_state.last_metadata = {}
            st.session_state["_img"] = None  # drop the held query image
            _reset_review()
            st.rerun()
        with st.expander("ℹ️ What does everything do?"):
            st.markdown(GUIDE_MARKDOWN)
            st.markdown(":green[● High ≥ 80%]  \n:orange[● Medium 50–79%]  \n:red[● Low < 50%]")

    mode = st.session_state.get("mode", MODE_SUGGEST)
    _render_header(backend)
    _render_mode_banner(mode)

    if st.session_state.pop("run_requested", False) and image is not None:
        with st.spinner("Traversing visual, text, and category signals…"):
            try:
                _reset_review()
                st.session_state.prediction = _run_prediction(backend, image, metadata, top_k, _session(backend))
                st.session_state.last_metadata = metadata
                # Keep exactly one query image so Accept/Reject can hand it to the engine.
                st.session_state["_img"] = image
            except Exception as exc:
                st.session_state.prediction = None
                st.error(f"The prediction could not be completed safely: {type(exc).__name__}.")

    if mode == MODE_SUGGEST:
        _page_suggest(image, st.session_state.prediction)
    elif mode == MODE_REVIEW:
        _page_review(st.session_state.prediction)
    else:
        _page_train(backend)


if __name__ == "__main__":
    main()

# Health check: launch from the repository root with `streamlit run app.py`.