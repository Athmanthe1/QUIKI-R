"""
Commons Category Suggester - Streamlit front-end (app.py)
=========================================================
Hackathon challenge: "Automatic Category Suggestion for Wikimedia Commons".

This file is the *entire* UI, written in pure Python / Streamlit (no JavaScript,
no raw HTML). It talks to ``engine.py`` (SigLIP/CLIP + kNN / text / metadata
scorers + stacking fusion + NetworkX hierarchy post-processing) through a small
adapter, and degrades gracefully to a built-in MOCK engine when ``engine.py``
is missing, broken, or still being written - so the UI can always be demoed.

HEALTH CHECK  (GitHub Codespaces terminal)
------------------------------------------
    pip install -r requirements.txt
    python app.py --selftest          # headless sanity check of the adapter + mock (no browser)
    streamlit run app.py              # then click the forwarded-port popup (port 8501)

If the sidebar shows "Mock mode", the UI is healthy but ``engine.py`` was not
loaded; the exact reason is displayed there. Press "Reload engine" after fixing it.

ENGINE CONTRACT (any ONE shape is auto-detected - see ENGINE_CONTRACT below)
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import math
import random
import re
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))  # so `import engine` works wherever Streamlit is launched

MAX_UPLOAD_MB = 10      # reject larger uploads (protects Codespaces RAM)
MAX_PIXELS = 40_000_000  # reject absurd resolutions before decoding (decompression-bomb guard)
MAX_TEXT_LEN = 2000     # cap pasted metadata so one giant paste cannot stall the engine
ALLOWED_EXTS = (".jpg", ".jpeg", ".png")
ALLOWED_FORMATS = ("JPEG", "PNG")
MAX_SIDE = 1024         # working copy of the upload is downscaled to this
THUMB_SIDE = 320        # neighbour thumbnails kept in memory at this size
MAX_K = 15              # engine is always asked for this many; sliders slice at render time
MAX_NEIGHBORS = 12
TAU = 0.07              # temperature of the similarity-weighted vote (matches s_knn in the spec)

MODE_AUTO = "Auto - use engine.py, fall back to mock"
MODE_MOCK = "Force mock mode (UI testing)"

ENGINE_CONTRACT = '''\
# engine.py - ANY of these shapes is auto-detected by app.py
#
#   class CategorySuggestionEngine:            # also: CategoryEngine, SuggestionEngine, Engine
#       def suggest(self, image, title="", description="", geotag="",
#                   top_k=10, n_neighbors=6) -> dict: ...
#   # or module-level: load_engine() / get_engine() returning such an object,
#   # or plain module functions: suggest_categories(...) / suggest(...) / predict(...)
#
# `image` is a PIL.Image (RGB). Parameter names are matched flexibly
# (image/img, title, description/caption, geotag/location, metadata dict, top_k/k ...).
#
# Return value (dict, or just the suggestions list):
{
  "suggestions": [
    {"category": "Bridges",              # name (a leading "Category:" is stripped)
     "confidence": 0.91,                 # 0-1 (values >1 are read as percent)
     "path": ["Architecture", "Structures by type", "Bridges"],   # root -> leaf, optional
     "scores": {"knn": 0.93, "txt": 0.71, "cls": 0.88, "meta": 0.10}}   # optional
  ],
  "neighbors": [                          # optional, powers the Explainability panel
    {"image": "<PIL.Image | file path | URL | bytes>",
     "similarity": 0.88,                  # cosine similarity
     "categories": ["Bridges", "Rivers"], # labels of that reference image
     "title": "File:Example.jpg"}
  ]
}
# Optional feedback hook:  record_feedback(category, accepted: bool) on the engine object.
'''

ALIASES: dict[str, set[str]] = {
    "image": {"image", "img", "pil_image", "pil_img", "query_image", "photo", "x"},
    "title": {"title", "image_title"},
    "description": {"description", "desc", "caption"},
    "geotag": {"geotag", "geo", "location", "coords", "coordinates", "gps"},
    "metadata": {"metadata", "meta", "md"},
    "top_k": {"top_k", "k", "topk", "num_suggestions", "limit"},
    "n_neighbors": {"n_neighbors", "k_neighbors", "num_neighbors", "n_neighbours", "knn_k"},
    "category": {"category", "label", "name"},
    "accepted": {"accepted", "accept", "is_accepted", "positive"},
    "verdict": {"verdict", "feedback", "action", "decision"},
}
CLASS_CANDIDATES = ("CategorySuggestionEngine", "CategoryEngine", "SuggestionEngine",
                    "CommonsEngine", "CommonsCategoryEngine", "Engine")
FACTORY_CANDIDATES = ("load_engine", "get_engine", "build_engine", "init_engine", "create_engine")
ENTRY_POINTS = ("suggest_categories", "suggest", "predict_categories", "predict",
                "run_inference", "infer", "run")


# --------------------------------------------------------------------------- #
# Mock engine (used when engine.py is unavailable)
# --------------------------------------------------------------------------- #
# (keywords, category, path root -> leaf)
MOCK_TAXONOMY = [
    (("temple", "church", "cathedral", "mosque", "shrine", "monastery"), "Religious buildings",
     ["Architecture", "Buildings by function", "Religious buildings"]),
    (("mountain", "peak", "hill", "alps", "himalaya"), "Mountains",
     ["Nature", "Landforms", "Mountains"]),
    (("beach", "sea", "ocean", "coast", "shore"), "Beaches",
     ["Nature", "Bodies of water", "Coasts", "Beaches"]),
    (("bridge", "viaduct"), "Bridges",
     ["Architecture", "Structures by type", "Bridges"]),
    (("bird", "owl", "eagle", "sparrow"), "Birds",
     ["Nature", "Animals", "Birds"]),
    (("flower", "rose", "blossom", "tulip"), "Flowers",
     ["Nature", "Plants", "Flowers"]),
    (("street", "market", "city", "downtown", "square"), "Urban streets",
     ["Geography", "Human settlements", "Urban streets"]),
    (("festival", "protest", "rally", "parade", "ceremony", "conference"), "Public events",
     ["Events", "Public events"]),
    (("food", "dish", "meal", "cuisine", "cake"), "Food and drink",
     ["Culture", "Food and drink"]),
    (("crowd", "group", "audience", "team"), "Groups of people",
     ["People", "Groups of people"]),
    (("car", "bus", "train", "truck", "railway"), "Road and rail vehicles",
     ["Transport", "Vehicles", "Road and rail vehicles"]),
    (("sunset", "sunrise", "sky", "cloud", "dusk"), "Sunsets and skies",
     ["Nature", "Sky", "Sunsets and skies"]),
    (("diagram", "chart", "map", "graph", "infographic"), "Diagrams and maps",
     ["Abstract concepts", "Information graphics", "Diagrams and maps"]),
    (("tree", "forest", "woodland", "oak"), "Trees",
     ["Nature", "Plants", "Trees"]),
    (("boat", "ship", "harbor", "harbour", "ferry"), "Boats and ships",
     ["Transport", "Watercraft", "Boats and ships"]),
]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mock_thumb(img: Image.Image, rng: random.Random, size: int = 256) -> Image.Image:
    """A fake 'reference image': a random crop of the query so the demo looks plausible."""
    w, h = img.size
    scale = rng.uniform(0.55, 1.0)
    cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
    x0, y0 = rng.randint(0, max(0, w - cw)), rng.randint(0, max(0, h - ch))
    crop = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((size, size))
    if rng.random() < 0.5:
        crop = ImageOps.mirror(crop)
    return ImageEnhance.Brightness(crop).enhance(rng.uniform(0.85, 1.15))


def _mock_suggest(image: Image.Image, title: str, description: str, geotag: str,
                  top_k: int, n_neighbors: int) -> dict:
    digest = hashlib.sha256(image.resize((32, 32)).tobytes()).hexdigest()
    rng = random.Random(int(digest[:12], 16))
    text = f"{title} {description} {geotag}".lower()

    def hit(keywords: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(k)}s?\b", text) for k in keywords)

    matched = [e for e in MOCK_TAXONOMY if hit(e[0])]
    matched_names = {e[1] for e in matched}
    rest = [e for e in MOCK_TAXONOMY if e[1] not in matched_names]
    rng.shuffle(rest)

    suggestions = []
    for keywords, name, path in (matched + rest)[:top_k]:
        is_hit = name in matched_names
        conf = rng.uniform(0.74, 0.97) if is_hit else rng.uniform(0.04, 0.50)
        suggestions.append({
            "category": name,
            "confidence": conf,
            "path": path,
            "scores": {
                "knn": _clip01(conf + rng.uniform(-0.10, 0.10)),
                "txt": _clip01(conf + rng.uniform(-0.15, 0.10)),
                "cls": _clip01(conf + rng.uniform(-0.10, 0.10)),
                "meta": _clip01((0.6 + rng.uniform(0, 0.3)) if is_hit else rng.uniform(0, 0.15)),
            },
        })
    suggestions.sort(key=lambda s: s["confidence"], reverse=True)
    names = [s["category"] for s in suggestions]

    neighbors = []
    for j in range(n_neighbors):
        cats = [names[0]] if rng.random() < 0.75 else []
        if len(names) > 1:
            cats.append(names[rng.randint(1, min(4, len(names) - 1))])
        neighbors.append({
            "image": _mock_thumb(image, rng),
            "similarity": max(0.40, 0.93 - 0.025 * j - rng.uniform(0, 0.02)),
            "categories": cats or [names[0]],
            "title": f"MOCK reference #{j + 1}",
        })
    return {"suggestions": suggestions, "neighbors": neighbors}


# --------------------------------------------------------------------------- #
# Flexible adapter: call whatever engine.py exposes, normalise whatever it returns
# --------------------------------------------------------------------------- #
def _first(obj: Any, keys: tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        val = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
        if val is not None:
            return val
    return default


def _clean_name(s: Any) -> str:
    return re.sub(r"^category:\s*", "", str(s).strip(), flags=re.I).replace("_", " ").strip()


def _norm_name(s: Any) -> str:
    return _clean_name(s).casefold()


def _to_path(value: Any, name: str) -> list[str]:
    parts: list[str] = []
    if isinstance(value, str):
        parts = [p for p in re.split(r"\s*(?:›|->|→|>|\|)\s*", value) if p.strip()]
    elif isinstance(value, (list, tuple)) and value:
        if isinstance(value[0], (list, tuple)):  # several paths -> take the first
            value = value[0]
        parts = [str(p) for p in value if str(p).strip()]
    parts = [_clean_name(p) for p in parts]
    if not parts or parts[-1].casefold() != name.casefold():
        parts.append(name)  # engine gave ancestors only (or nothing)
    return parts


def _load_display_image(ref: Any):
    """Return a small PIL image, a URL string, or None. Never raises."""
    try:
        if isinstance(ref, Image.Image):
            img = ref.copy()
        elif isinstance(ref, (bytes, bytearray)):
            img = Image.open(io.BytesIO(ref))
        elif isinstance(ref, (str, Path)):
            s = str(ref)
            if s.lower().startswith(("http://", "https://")):
                return s
            if not Path(s).is_file():
                return None
            img = Image.open(s)
        else:
            return None
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((THUMB_SIDE, THUMB_SIDE))
        return img
    except Exception:
        return None


def _normalize_suggestion(item: Any) -> dict:
    if isinstance(item, (tuple, list)) and item and isinstance(item[0], str):
        item = {"category": item[0],
                "confidence": item[1] if len(item) > 1 else 0.0,
                "path": item[2] if len(item) > 2 else None}
    name = _clean_name(_first(item, ("category", "name", "label", "title", "id"), "(unnamed)"))
    try:
        conf = float(_first(item, ("confidence", "score", "probability", "prob", "p", "fused_score"), 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    if not math.isfinite(conf):  # NaN / inf from a misbehaving scorer must not render as 100%
        conf = 0.0
    if conf > 1.0:
        conf = conf / 100.0
    scores_raw = _first(item, ("scores", "components", "scorers", "breakdown"), {})
    scores = {}
    if isinstance(scores_raw, dict):
        for k, v in scores_raw.items():
            try:
                scores[re.sub(r"^s_", "", str(k).lower())] = float(v)
            except (TypeError, ValueError):
                pass
    path = _first(item, ("path", "hierarchy", "lineage", "ancestors", "dag_path", "breadcrumb"))
    return {"name": name, "conf": _clip01(conf), "path": _to_path(path, name), "scores": scores}


def _normalize_neighbor(item: Any, idx: int) -> dict:
    if isinstance(item, (str, bytes, bytearray, Path, Image.Image)):
        item = {"image": item}
    ref = _first(item, ("image", "img", "pil_image", "thumbnail", "thumb", "image_path",
                        "path", "filepath", "file", "url", "uri"))
    sim = _first(item, ("similarity", "sim", "cosine", "score"))
    try:
        sim = float(sim) if sim is not None else None
    except (TypeError, ValueError):
        sim = None
    if sim is not None and not math.isfinite(sim):
        sim = None
    cats = _first(item, ("categories", "labels", "category", "tags"), [])
    if isinstance(cats, str):
        cats = [cats]
    elif not isinstance(cats, (list, tuple, set)):
        try:
            cats = list(cats)
        except TypeError:
            cats = [cats]
    return {
        "display": _load_display_image(ref),
        "sim": sim,
        "categories": [_clean_name(c) for c in cats if c is not None and str(c).strip()],
        "title": str(_first(item, ("title", "name", "caption"), f"Reference #{idx + 1}")),
    }


def normalize_result(raw: Any, top_k: int, n_neighbors: int) -> dict:
    """Accept the shapes listed in ENGINE_CONTRACT (and a few near-misses)."""
    sug_raw, nb_raw = None, []
    if isinstance(raw, tuple) and len(raw) == 2 and all(isinstance(p, (list, tuple)) for p in raw):
        sug_raw, nb_raw = raw
    elif isinstance(raw, (list, tuple)):
        sug_raw = raw
    elif raw is not None:
        sug_raw = _first(raw, ("suggestions", "categories", "predictions", "results", "ranked"))
        nb_raw = _first(raw, ("neighbors", "neighbours", "nearest_neighbors", "nearest_neighbours",
                              "references", "explanations"), [])
    if isinstance(sug_raw, dict):  # {"Bridges": 0.9, ...}
        sug_raw = [{"category": k, "confidence": v} for k, v in sug_raw.items()]
    if sug_raw is None:
        raise ValueError("Engine output not recognised - expected a list of suggestions or a dict "
                         "with a 'suggestions' key (see the engine contract in the footer).")
    suggestions = sorted((_normalize_suggestion(s) for s in sug_raw),
                         key=lambda s: s["conf"], reverse=True)[:top_k]
    neighbors = [_normalize_neighbor(n, i) for i, n in enumerate(list(nb_raw or [])[:n_neighbors])]
    return {"suggestions": suggestions, "neighbors": neighbors}


def _call_flexibly(fn, values: dict[str, Any]):
    """Call fn, binding our values to whatever parameter names it declares."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(values["image"]) if "image" in values else fn()
    kwargs, image_bound = {}, False
    for pname, p in params.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        for canon, names in ALIASES.items():
            if pname.lower() in names and canon in values:
                kwargs[pname] = values[canon]
                image_bound = image_bound or canon == "image"
                break
    if "image" in values and not image_bound:
        return fn(values["image"], **kwargs)
    return fn(**kwargs)


class Backend:
    """Uniform wrapper around engine.py (real) or the built-in mock."""

    def __init__(self, name: str, is_mock: bool, impl: Any = None):
        self.name, self.is_mock, self.impl = name, is_mock, impl

    @classmethod
    def mock(cls) -> "Backend":
        return cls("Mock engine (demo data)", True)

    def entry_point(self):
        for n in ENTRY_POINTS:
            fn = getattr(self.impl, n, None)
            if callable(fn):
                return fn
        return None

    def suggest(self, image: Image.Image, title: str, description: str, geotag: str,
                top_k: int, n_neighbors: int) -> dict:
        t0 = time.perf_counter()
        if self.is_mock:
            raw = _mock_suggest(image, title, description, geotag, top_k, n_neighbors)
        else:
            fn = self.entry_point()
            if fn is None:
                raise AttributeError(f"engine exposes none of: {', '.join(ENTRY_POINTS)}")
            raw = _call_flexibly(fn, {
                "image": image, "title": title, "description": description, "geotag": geotag,
                "metadata": {"title": title, "description": description, "geotag": geotag},
                "top_k": top_k, "n_neighbors": n_neighbors,
            })
        result = normalize_result(raw, top_k, n_neighbors)
        result.update(latency_ms=(time.perf_counter() - t0) * 1000, backend=self.name,
                      is_mock=self.is_mock)
        return result

    def send_feedback(self, category: str, verdict: str) -> None:
        """Forward accept/reject to engine.record_feedback if it exists (best effort)."""
        hook = None if self.is_mock else getattr(self.impl, "record_feedback", None)
        if callable(hook):
            try:
                _call_flexibly(hook, {"category": category, "verdict": verdict,
                                      "accepted": verdict == "accept"})
            except Exception:
                pass


def _instantiate(mod: Any) -> Any:
    for cls_name in CLASS_CANDIDATES:
        cls = getattr(mod, cls_name, None)
        if inspect.isclass(cls):
            return cls()
    for fac in FACTORY_CANDIDATES:
        f = getattr(mod, fac, None)
        if callable(f) and not inspect.isclass(f):
            return f()
    return mod  # module-level functions


@st.cache_resource(show_spinner="Loading engine.py (first run only)...")
def load_real_backend() -> tuple[Backend | None, str]:
    """Import + initialise engine.py exactly once per server process (heavy models live here)."""
    importlib.invalidate_caches()    # lets "Reload engine" notice an engine.py created after startup
    sys.modules.pop("engine", None)  # forget any half-imported copy from a previous failed attempt
    try:
        mod = importlib.import_module("engine")
    except ModuleNotFoundError as exc:
        if exc.name == "engine":
            return None, "engine.py not found next to app.py."
        return None, f"engine.py needs a missing package: {exc.name}."
    except (Exception, SystemExit) as exc:  # syntax errors, CUDA problems, half-written file, sys.exit()...
        return None, f"engine.py failed to import: {exc!r}"
    try:
        impl = _instantiate(mod)
        backend = Backend(f"engine.{type(impl).__name__ if impl is not mod else 'py (functions)'}",
                          False, impl)
        if backend.entry_point() is None:
            return None, f"engine.py has no recognised entry point ({', '.join(ENTRY_POINTS)})."
    except (Exception, SystemExit) as exc:
        return None, f"engine.py loaded but failed to initialise: {exc!r}"
    return backend, "Connected to engine.py."


def resolve_backend(mode: str) -> tuple[Backend, str, bool]:
    """-> (backend, status message, is_real)."""
    if mode == MODE_MOCK:
        return Backend.mock(), "Mock mode forced from the sidebar.", False
    real, msg = load_real_backend()
    if real is not None:
        return real, msg, True
    return Backend.mock(), msg, False


# --------------------------------------------------------------------------- #
# Input guards (pure functions, no Streamlit calls - exercised by --selftest)
# --------------------------------------------------------------------------- #
class ImageInputError(ValueError):
    """An upload that cannot be used. The message is written to be shown to the user as-is."""


def load_image_bytes(data: bytes, filename: str = "upload") -> Image.Image:
    """Validate + decode an uploaded image. Raises ImageInputError with a friendly message."""
    ext = Path(filename or "").suffix.lower()
    if ext and ext not in ALLOWED_EXTS:
        raise ImageInputError(f"'{filename}' is not a JPG or PNG image. "
                              "Please upload a .jpg, .jpeg or .png file.")
    if not data:
        raise ImageInputError("The uploaded file is empty (0 bytes).")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ImageInputError(f"The file is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller image.")
    try:
        probe = Image.open(io.BytesIO(data))  # lazy: reads the header only
        if probe.format not in ALLOWED_FORMATS:
            raise ImageInputError(f"The file content is {probe.format or 'unknown'}, not JPEG or PNG - "
                                  "was it renamed from another format?")
        if probe.width * probe.height > MAX_PIXELS:
            raise ImageInputError(f"The image is {probe.width}x{probe.height}px, which is too large to "
                                  "process safely. Please downscale it first.")
        probe.verify()  # cheap structural check (consumes the handle)
        img = Image.open(io.BytesIO(data))
        if img.format == "JPEG":
            img.draft("RGB", (MAX_SIDE, MAX_SIDE))  # decode big JPEGs at reduced size to save RAM
        img.load()  # full decode: catches truncated / corrupt pixel data
        img = ImageOps.exif_transpose(img).convert("RGB")
    except ImageInputError:
        raise
    except UnidentifiedImageError:
        raise ImageInputError("This file is not a readable image. It may be a text/PDF file renamed "
                              "to .jpg/.png, or it may be corrupted.") from None
    except Image.DecompressionBombError:
        raise ImageInputError("The image has an extreme pixel count and was rejected for safety.") from None
    except (OSError, SyntaxError, ValueError, EOFError, MemoryError):
        raise ImageInputError("The image appears damaged or truncated and could not be decoded. "
                              "Please re-export it and try again.") from None
    except Exception:
        raise ImageInputError("The image could not be processed. Please try a different file.") from None
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_COORD_COMMA = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*[,;]\s*([+-]?\d+(?:\.\d+)?)\s*$")
_COORD_SPACE = re.compile(r"^\s*([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s*$")


def clean_text(value: Any, max_len: int = MAX_TEXT_LEN) -> str:
    """Always returns a plain str: None -> '', bytes decoded, control chars stripped, length capped."""
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", "replace")
    return _CTRL_CHARS.sub("", str(value)).strip()[:max_len]


def parse_geotag(raw: Any) -> tuple[str, str | None]:
    """-> (safe geotag string, optional note). Out-of-range 'lat, lon' pairs are dropped with a note."""
    text = clean_text(raw, 200)
    m = _COORD_COMMA.match(text) or _COORD_SPACE.match(text)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return "", (f"The geotag '{text}' is not a valid latitude/longitude pair "
                        "(latitude -90..90, longitude -180..180), so it was ignored.")
        return f"{lat:.6f}, {lon:.6f}", None
    return text, None  # free-text place names pass through untouched


# --------------------------------------------------------------------------- #
# Session-state & UI guards
# --------------------------------------------------------------------------- #
REQUIRED_RESULT_KEYS = ("suggestions", "neighbors", "query_thumb", "backend", "is_mock", "latency_ms")


def init_state() -> None:
    """Every key the app reads gets a safe default once, so widgets never hit a missing key."""
    st.session_state.setdefault("feedback", {})
    st.session_state.setdefault("engine_gen", 0)      # bumped by "Reload engine"
    st.session_state.setdefault("last_request", None)  # fingerprint of the last completed run


def reset_image_state() -> None:
    for k in ("_img_key", "_img", "result", "last_request"):
        st.session_state.pop(k, None)
    st.session_state["feedback"] = {}


def get_result() -> dict | None:
    """The stored result, or None if it is missing, malformed, or belongs to a different image."""
    r = st.session_state.get("result")
    if (not isinstance(r, dict) or any(k not in r for k in REQUIRED_RESULT_KEYS)
            or r.get("image_key") != st.session_state.get("_img_key")):
        st.session_state.pop("result", None)
        return None
    return r


@contextmanager
def guarded(section: str):
    """Turn an unexpected error inside one panel into a message; the rest of the page keeps working."""
    try:
        yield
    except Exception as exc:  # Streamlit's rerun/stop signals are BaseExceptions and pass through
        st.error(f"The {section} hit an unexpected problem ({type(exc).__name__}: {exc}). "
                 "The rest of the app is still usable - try again or reload the page.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc(), language="text")


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def show_image(img: Any, caption: str | None = None) -> None:
    """st.image that works across Streamlit versions (width API changed in 1.5x)."""
    try:
        st.image(img, caption=caption, width="stretch")
    except Exception:
        st.image(img, caption=caption, use_container_width=True)


def conf_color(pct: float) -> str:
    return "green" if pct >= 70 else "orange" if pct >= 40 else "red"


def get_uploaded_image(uploaded) -> Image.Image:
    """Validate + decode once per upload. Successes AND failures are cached in session_state,
    so reruns (slider moves, button clicks) never re-decode a file or re-show a stale result."""
    key = (uploaded.name, uploaded.size)
    if st.session_state.get("_img_key") == key and st.session_state.get("_img") is not None:
        return st.session_state["_img"]
    cached_err = st.session_state.get("_img_err")
    if cached_err and cached_err[0] == key:
        raise ImageInputError(cached_err[1])
    try:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            raise ImageInputError(f"The file is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller image.")
        img = load_image_bytes(uploaded.getvalue(), uploaded.name)
    except ImageInputError as exc:
        reset_image_state()  # never keep showing results that belong to a previous image
        st.session_state["_img_err"] = (key, str(exc))
        raise
    reset_image_state()  # new image -> old suggestions and feedback are stale
    st.session_state.update(_img_key=key, _img=img, _img_err=None)
    return img


def record_feedback(category: str, verdict: str) -> None:
    st.session_state.setdefault("feedback", {})[category] = verdict
    backend = st.session_state.get("_backend")
    if backend is not None:
        backend.send_feedback(category, verdict)


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def render_sidebar() -> dict:
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Backend mode", [MODE_AUTO, MODE_MOCK], key="mode")
        top_k = st.slider("Suggestions shown (k)", 3, MAX_K, 8, key="top_k")
        n_neighbors = st.slider("Reference neighbours", 2, MAX_NEIGHBORS, 6, key="n_neighbors")
        min_conf = st.slider("Minimum confidence (%)", 0, 90, 0, step=5, key="min_conf")
        show_hierarchy = st.toggle("Show DAG hierarchy path", value=True, key="show_hierarchy")

        st.divider()
        backend, status, is_real = resolve_backend(mode)
        st.session_state["_backend"] = backend
        if is_real:
            st.success(f"Live engine: {backend.name}")
        else:
            st.warning("Mock mode active - see the notice at the top of the page.")
        if st.button("Reload engine"):
            load_real_backend.clear()
            st.session_state["engine_gen"] = st.session_state.get("engine_gen", 0) + 1
            st.rerun()
        st.caption(f"Feedback logged this session: {len(st.session_state.get('feedback', {}))}")
    return {"backend": backend, "status": status, "is_real": is_real, "top_k": top_k,
            "n_neighbors": n_neighbors, "min_conf": min_conf, "show_hierarchy": show_hierarchy}


def run_inference(image: Image.Image, title: str, description: str, geotag: str, cfg: dict,
                  notes: list[str] | None = None) -> None:
    backend: Backend = cfg["backend"]
    # Rapid-fire guard: an identical resubmit (double-click, repeated Enter) reuses the stored result.
    request = (st.session_state.get("_img_key"), title, description, geotag, backend.name,
               st.session_state.get("engine_gen", 0))
    existing = get_result()
    if (existing is not None and not existing.get("fallback_reason")
            and st.session_state.get("last_request") == request):
        getattr(st, "toast", st.info)("Same image and metadata - showing the existing results.")
        return
    fallback_reason = None
    try:
        with st.spinner("Encoding image and searching the index..."):
            try:
                result = backend.suggest(image, title, description, geotag, MAX_K, MAX_NEIGHBORS)
            except Exception as exc:
                if backend.is_mock:
                    raise
                fallback_reason = f"{type(exc).__name__}: {exc}"
                st.session_state["last_traceback"] = traceback.format_exc()
                result = Backend.mock().suggest(image, title, description, geotag, MAX_K, MAX_NEIGHBORS)
        thumb = image.copy()
        thumb.thumbnail((512, 512))
        result.update(query_thumb=thumb, fallback_reason=fallback_reason, notes=list(notes or []),
                      image_key=st.session_state.get("_img_key"))
    except Exception as exc:
        st.error(f"Could not generate suggestions ({type(exc).__name__}: {exc}). "
                 "Your previous results, if any, are unchanged - please try again.")
        return
    # Commit only after everything succeeded: a failed or interrupted run never wipes the dashboard.
    st.session_state["result"] = result
    st.session_state["last_request"] = request
    st.session_state["feedback"] = {}


def render_input_column(cfg: dict) -> None:
    st.subheader("Input gateway")
    uploaded = st.file_uploader(
        "Drag & drop an image (JPG / PNG)", type=["jpg", "jpeg", "png"],
        help=f"Max {MAX_UPLOAD_MB} MB. Large images are downscaled to {MAX_SIDE}px to save memory.",
    )
    image, upload_failed = None, False
    if uploaded is None:
        reset_image_state()
        st.session_state.pop("_img_err", None)
    else:
        try:
            image = get_uploaded_image(uploaded)
        except ImageInputError as exc:
            upload_failed = True
            st.error(f"Invalid upload: {exc}")
        except Exception as exc:  # anything unexpected while decoding - still no raw traceback
            upload_failed = True
            st.error("This image could not be read. Please try a different JPG or PNG file.")
            st.caption(f"Technical detail: {type(exc).__name__}")
    if image is not None:
        show_image(image, caption=f"{uploaded.name} - working copy {image.width}x{image.height}px")

    with st.form("metadata_form"):
        title = st.text_input("Image title", placeholder="e.g. Golden Gate Bridge at sunset")
        description = st.text_area("Description", height=100,
                                   placeholder="Caption, event, people-free context, date ...")
        geotag = st.text_input("Geotag", placeholder="e.g. 37.8199, -122.4783  or  San Francisco, USA")
        submitted = st.form_submit_button("Suggest categories", type="primary")
    if submitted:
        if image is None:
            st.warning("The uploaded file is not usable - please fix it above and try again."
                       if upload_failed else "Please upload an image first.")
        else:
            # Ghost-metadata guard: blank / whitespace / None fields always reach the engine as safe strings.
            title, description = clean_text(title), clean_text(description)
            geotag, geo_note = parse_geotag(geotag)
            notes = [geo_note] if geo_note else []
            if not (title or description or geotag):
                notes.append("No metadata was provided, so these suggestions rely on the image alone. "
                             "A title, description or geotag usually helps with events and places.")
            run_inference(image, title, description, geotag, cfg, notes)


def render_suggestion_card(i: int, s: dict, cfg: dict) -> None:
    pct = s["conf"] * 100
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{i + 1}. {s['name']}**")
        c2.markdown(f":{conf_color(pct)}[**{pct:.1f}%**]")
        st.progress(s["conf"])
        if cfg["show_hierarchy"]:
            parts = s["path"]
            if len(parts) > 1:
                st.caption("Path: " + " › ".join(parts[:-1]) + f" › **{parts[-1]}**")
            else:
                st.caption("Path: top-level category")
        if s["scores"]:
            labels = {"knn": "kNN", "txt": "text", "cls": "classifier", "meta": "metadata"}
            st.caption("Scorers: " + " · ".join(f"{labels.get(k, k)} {v:.2f}" for k, v in s["scores"].items()))
        verdict = st.session_state.get("feedback", {}).get(s["name"])
        b1, b2, b3 = st.columns([1, 1, 3])
        b1.button("Accept", key=f"acc_{i}", on_click=record_feedback, args=(s["name"], "accept"),
                  disabled=verdict == "accept")
        b2.button("Reject", key=f"rej_{i}", on_click=record_feedback, args=(s["name"], "reject"),
                  disabled=verdict == "reject")
        if verdict:
            b3.caption(f"Marked: {verdict}")


def render_results_column(cfg: dict) -> list[dict]:
    st.subheader("Results dashboard")
    result = get_result()
    if not result:
        st.info("Upload an image and press **Suggest categories** to see ranked suggestions here.")
        return []
    for note in result.get("notes", []):
        st.info(note)
    if result.get("fallback_reason"):
        st.warning("engine.py raised an error, so these results are MOCK data. "
                   f"Reason: {result['fallback_reason']}")
        with st.expander("Technical details"):
            st.code(st.session_state.get("last_traceback", ""), language="text")
    elif result["is_mock"]:
        st.warning("Mock mode: demo data only - these are not real model predictions.")

    sugs = [s for s in result["suggestions"][:cfg["top_k"]] if s["conf"] * 100 >= cfg["min_conf"]]
    st.caption(f"Backend: {result['backend']} · latency {result['latency_ms']:.0f} ms · "
               f"{len(sugs)} suggestion(s) shown. Suggestions are for human review only.")
    if not sugs:
        st.info("No suggestion passes the confidence threshold - lower it in the sidebar.")
    for i, s in enumerate(sugs):
        render_suggestion_card(i, s, cfg)
    return sugs


def render_explainability(cfg: dict, sugs: list[dict]) -> None:
    result = get_result()
    st.divider()
    st.subheader("Explainability panel - why did the AI suggest this?")
    if not result:
        st.caption("Nearest-neighbour reference images will appear here after a run.")
        return
    st.caption("These are the closest already-categorised images retrieved from the vector database. "
               "A suggestion is trustworthy when many similar references carry the same category.")
    neighbors = result["neighbors"][:cfg["n_neighbors"]]
    if not sugs:
        return
    if not neighbors:
        st.info("The engine returned no neighbour images, so there is nothing to show yet.")
        return

    labels = [f"{i + 1}. {s['name']} ({s['conf'] * 100:.0f}%)" for i, s in enumerate(sugs)]
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.markdown("**Query image**")
        show_image(result["query_thumb"])
        if st.session_state.get("explain_sel") not in labels:
            st.session_state.pop("explain_sel", None)
        sel = st.selectbox("Explain suggestion", labels, key="explain_sel")
        chosen = sugs[labels.index(sel)]
        target = _norm_name(chosen["name"])
        supports = [target in {_norm_name(c) for c in nb["categories"]} for nb in neighbors]
        st.metric("Neighbours filed under it", f"{sum(supports)} / {len(neighbors)}")
        sims = [nb["sim"] for nb in neighbors]
        if all(v is not None for v in sims):
            top = max(sims)
            w = [math.exp((v - top) / TAU) for v in sims]
            share = sum(wi for wi, ok in zip(w, supports) if ok) / sum(w)
            st.metric("Similarity-weighted vote", f"{share * 100:.0f}%")
    with right:
        per_row = 4
        for start in range(0, len(neighbors), per_row):
            cols = st.columns(per_row)
            for col, nb, ok in zip(cols, neighbors[start:start + per_row], supports[start:start + per_row]):
                with col:
                    if nb["display"] is not None:
                        show_image(nb["display"])
                    else:
                        st.caption("(no preview available)")
                    sim_txt = f"similarity {nb['sim']:.2f}" if nb["sim"] is not None else "similarity n/a"
                    st.caption(f"{nb['title']} · {sim_txt}")
                    if nb["categories"]:
                        st.caption("Filed under: " + ", ".join(nb["categories"][:3]))
                    st.markdown(":green[Supports this suggestion]" if ok else ":gray[Not filed here]")


def render_footer() -> None:
    with st.expander("Health check & engine contract"):
        st.markdown("**Launch in the GitHub Codespaces terminal**")
        st.code("pip install -r requirements.txt\n"
                "python app.py --selftest     # headless sanity check\n"
                "streamlit run app.py         # open the forwarded port 8501", language="bash")
        st.markdown("**What app.py expects from engine.py**")
        st.code(ENGINE_CONTRACT, language="python")


def main() -> None:
    st.set_page_config(page_title="Commons Category Suggester", page_icon="🗂️",
                       layout="wide", initial_sidebar_state="expanded")
    init_state()
    st.title("🗂️ Automatic Category Suggestion for Wikimedia Commons")
    st.caption("Upload an image, add optional metadata, and review ranked category suggestions "
               "with their hierarchy paths. Nothing is edited automatically - a human decides.")
    cfg = render_sidebar()
    if not cfg["is_real"]:
        st.warning(f"Running in MOCK mode: {cfg['status']} The interface is fully usable, but the "
                   "suggestions below are demo data, not real model output.")
    sugs: list[dict] = []
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        with guarded("input panel"):
            render_input_column(cfg)
    with right:
        with guarded("results dashboard"):
            sugs = render_results_column(cfg)
    with guarded("explainability panel"):
        render_explainability(cfg, sugs)
    render_footer()


# --------------------------------------------------------------------------- #
# Headless self-test:  python app.py --selftest
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    img = Image.new("RGB", (320, 240), (40, 120, 200))
    res = Backend.mock().suggest(img, "Golden Gate Bridge at sunset", "Suspension bridge over the bay",
                                 "37.8199, -122.4783", 5, 4)
    sugs = res["suggestions"]
    assert 1 <= len(sugs) <= 5, "mock returned wrong number of suggestions"
    assert all(0.0 <= s["conf"] <= 1.0 for s in sugs), "confidence out of range"
    assert [s["conf"] for s in sugs] == sorted((s["conf"] for s in sugs), reverse=True), "not ranked"
    assert len(res["neighbors"]) == 4 and all(n["display"] is not None for n in res["neighbors"])
    assert any(s["name"] == "Bridges" for s in sugs), "keyword metadata should surface 'Bridges'"
    alt = normalize_result({"categories": {"Category:Bridges": 87, "Rivers": 0.2}}, 5, 3)
    assert abs(alt["suggestions"][0]["conf"] - 0.87) < 1e-9 and alt["suggestions"][0]["name"] == "Bridges"
    alt2 = normalize_result([("Trees", 0.5, "Nature > Plants")], 5, 3)
    assert alt2["suggestions"][0]["path"] == ["Nature", "Plants", "Trees"]
    # --- input guards: bad uploads ---
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buf, "PNG")
    png = buf.getvalue()
    assert load_image_bytes(png, "ok.png").mode == "RGB"

    def rejects(data: bytes, name: str) -> bool:
        try:
            load_image_bytes(data, name)
        except ImageInputError:
            return True
        return False

    assert rejects(b"hello world", "notes.txt"), "a .txt must be rejected"
    assert rejects(b"%PDF-1.4 not really", "doc.jpg"), "wrong content behind an image extension"
    assert rejects(png[: len(png) // 2], "cut.png"), "truncated PNG must be rejected"
    assert rejects(b"", "empty.png"), "empty file must be rejected"
    # --- input guards: blank / odd metadata ---
    assert clean_text(None) == "" and clean_text("  hi\x00 ") == "hi" and len(clean_text("x" * 5000)) == MAX_TEXT_LEN
    assert parse_geotag(None) == ("", None) and parse_geotag("") == ("", None)
    assert parse_geotag("37.8199, -122.4783") == ("37.819900, -122.478300", None)
    assert parse_geotag("91, 10")[0] == "" and parse_geotag("91, 10")[1], "out-of-range coords must be dropped"
    assert parse_geotag("Paris, France") == ("Paris, France", None)
    blank = Backend.mock().suggest(img, "", "", "", 5, 3)
    assert blank["suggestions"] and len(blank["neighbors"]) == 3, "blank metadata must still work"
    # --- adapter edge cases: NaN scores, odd label types ---
    odd = normalize_result({"suggestions": [{"category": "X", "confidence": float("nan")}],
                            "neighbors": [{"image": None, "similarity": float("nan"), "categories": 7}]}, 5, 3)
    assert odd["suggestions"][0]["conf"] == 0.0 and odd["neighbors"][0]["sim"] is None
    print("app.py selftest OK - adapter, input guards and mock engine are healthy.")


if "--selftest" in sys.argv:
    _selftest()
else:
    main()