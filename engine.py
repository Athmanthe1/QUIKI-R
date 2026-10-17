"""
engine.py - V1.4 serving engine.

* ``CategoryEngine`` is READ-ONLY after construction: one instance can be shared by every visitor
  (``@st.cache_resource``). Everything a visitor changes - feedback, accepted-image overlay index, the
  fine-tuned head - lives in that visitor's ``Session`` object.
* Scorers (kNN / text / metadata / head) come from scoring.py; fusion (softmax kNN vote, SigLIP logit
  scale/bias, logistic stacking with missing-scorer flags) from fusion.py; hierarchy post-processing
  from dag.py. Probabilities are returned as they are: no min-max rescaling.
* The state is explicit: ``engine.status['mode']`` is 'artifacts' (real data), 'demo' (50 toy categories)
  and ``load_error`` says why artifacts were not used. The UI banner reads this, it does not assume.
* torch is only imported when a head is trained (``train_supervised_head``) or the real SigLIP encoder
  is built, so everything else runs - and is tested - without it.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from PIL import Image

from dag import load_category_graph
from encoders import get_encoder, make_text_prototypes
from feedback import (FeedbackStore, SessionOverlay, append_event, f1_labelled, holdout_split,
                      image_id_from_embedding, iterate_minibatches, replay_sample)
from manifest import MODEL_NAME
from protocache import load_or_compute_prototypes
from scoring import Knowledge, load_knowledge, rank_categories

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("CommonsEngine")

DEFAULT_CATEGORIES = [
    "Mountains", "Rivers", "Lakes", "Forests", "Beaches", "Sunsets", "Skies", "Waterfalls",
    "Birds", "Cats", "Dogs", "Horses", "Insects", "Fish", "Flowers", "Trees",
    "Buildings", "Bridges", "Towers", "Castles", "Churches", "Skyscrapers", "Stadiums",
    "Cars", "Bicycles", "Trains", "Airplanes", "Ships", "Motorcycles",
    "Portraits", "Crowds", "Children", "Musicians", "Dancers",
    "Festivals", "Protests", "Weddings", "Sports", "Paintings", "Sculptures", "Diagrams", "Maps",
    "Dishes", "Fruits", "Computers", "Cameras", "Smartphones", "Nature", "Animals", "Architecture",
]
DEFAULT_HIERARCHY = [
    ("Nature", "Mountains"), ("Nature", "Rivers"), ("Nature", "Forests"), ("Nature", "Beaches"),
    ("Nature", "Animals"), ("Nature", "Flowers"),
    ("Animals", "Birds"), ("Animals", "Cats"), ("Animals", "Dogs"),
    ("Architecture", "Buildings"), ("Architecture", "Bridges"), ("Buildings", "Skyscrapers"),
    ("Transport", "Cars"), ("Transport", "Trains"), ("Transport", "Airplanes"),
    ("Events", "Festivals"), ("Events", "Sports"),
]


class Session:
    """Per-visitor mutable state. Never shared between visitors (keep it in st.session_state)."""

    def __init__(self, embed_dim: int, log_path: Optional[Union[str, Path]] = None):
        self.id = uuid.uuid4().hex[:12]
        self.feedback = FeedbackStore()
        self.overlay = SessionOverlay(embed_dim)
        self.head: Optional[Tuple[np.ndarray, np.ndarray]] = None      # fine-tuned copy (W, b)
        self.head_trained = False
        self.log_path = Path(log_path) if log_path else None            # append-only JSONL
        self._emb_cache: Dict[str, np.ndarray] = {}

    def counts(self) -> Dict[str, int]:
        return self.feedback.counts()


class CategoryEngine:
    def __init__(self, model_name: str = MODEL_NAME, encoder=None, artifact_dir: Optional[Union[str, Path]] = None,
                 load_default: bool = True, cache_dir: Union[str, Path] = ".cache/quiki"):
        logger.info("Initialising engine (encoder=%s)...", getattr(encoder, "model_name", model_name))
        self.encoder = encoder if encoder is not None else get_encoder()
        self.embed_dim = int(self.encoder.embed_dim)
        self.cache_dir = Path(cache_dir)
        self.knowledge: Optional[Knowledge] = None
        self.status: Dict[str, Any] = {"mode": "empty", "detail": "no knowledge base loaded", "load_error": None}
        if artifact_dir is not None:
            try:
                self.load_production_artifacts(artifact_dir)
            except Exception as exc:                   # never crash the UI; remember why, and say so in the banner
                self.status["load_error"] = f"{type(exc).__name__}: {exc}"
                logger.warning("artifact load failed: %s", self.status["load_error"], exc_info=True)
        if self.knowledge is None and load_default:    # only when needed: no wasted startup work
            self._load_default_knowledge_base()

    # ------------------------------------------------------------------ knowledge
    def load_production_artifacts(self, artifact_dir: Union[str, Path]) -> None:
        """Build a Knowledge from ``<artifact_dir>/manifest.json`` and swap it in atomically."""
        kn = load_knowledge(artifact_dir, encoder=self.encoder, expected_dim=self.embed_dim)
        self.knowledge = kn                                      # swap in only after everything loaded
        self.status = {"mode": "artifacts", "load_error": None,
                       "detail": f"{len(kn.classes)} classes, {kn.index.ntotal} indexed images ({kn.index_type})"}
        logger.info("Production artifacts loaded: %s", self.status["detail"])

    def _load_default_knowledge_base(self) -> None:
        graph = nx.DiGraph()
        for parent, child in DEFAULT_HIERARCHY:
            graph.add_edge(parent, child)
        protos = load_or_compute_prototypes(self.cache_dir / "demo", DEFAULT_CATEGORIES,
                                            lambda n: make_text_prototypes(self.encoder, n),
                                            getattr(self.encoder, "model_name", MODEL_NAME))
        self.knowledge = Knowledge(
            classes=list(DEFAULT_CATEGORIES), prototypes=protos, index=None, graph=graph, source="demo",
            logit_scale=float(getattr(self.encoder, "logit_scale", np.log(10))),
            logit_bias=float(getattr(self.encoder, "logit_bias", -5.0)),
            notes=["demo knowledge base: 50 toy categories, no image index, no trained head"])
        prev_error = self.status.get("load_error")
        self.status = {"mode": "demo", "load_error": prev_error,
                       "detail": "50 built-in toy categories; no image index, no trained head"}

    # ---- read-only views used by the UI ----
    @property
    def category_graph(self) -> nx.DiGraph:
        return self.knowledge.graph if self.knowledge else nx.DiGraph()

    @property
    def category_map(self) -> Dict[int, str]:
        return dict(enumerate(self.knowledge.classes)) if self.knowledge else {}

    @property
    def category_inv_map(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.knowledge.classes)} if self.knowledge else {}

    def state_summary(self) -> Dict[str, Any]:
        kn = self.knowledge
        return {"mode": self.status["mode"], "detail": self.status["detail"], "load_error": self.status.get("load_error"),
                "n_classes": len(kn.classes) if kn else 0, "n_vectors": int(kn.index.ntotal) if kn and kn.index is not None else 0,
                "index_type": kn.index_type if kn else "none", "calibrated": bool(kn and kn.stacker is not None),
                "has_head": bool(kn and kn.head is not None), "notes": list(kn.notes) if kn else []}

    def new_session(self, log_path: Optional[Union[str, Path]] = None) -> Session:
        return Session(self.embed_dim, log_path)

    # ------------------------------------------------------------------ embedding helpers
    @staticmethod
    def _to_pil(image_input: Any) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, (str, os.PathLike)):
            return Image.open(str(image_input)).convert("RGB")
        if isinstance(image_input, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(image_input))).convert("RGB")
        if hasattr(image_input, "read"):
            return Image.open(image_input).convert("RGB")
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def embed_image(self, image_input: Any) -> np.ndarray:
        return np.asarray(self.encoder.encode_images([self._to_pil(image_input)])[0], dtype=np.float32)

    def _meta_embedding(self, metadata: Optional[Dict]) -> Optional[np.ndarray]:
        if not isinstance(metadata, dict):
            return None
        text = f"{metadata.get('title') or ''}. {metadata.get('description') or ''}".strip(" .")
        return make_text_prototypes(self.encoder, [text])[0] if text else None

    # ------------------------------------------------------------------ inference
    def suggest(self, image: Any, metadata: Optional[Dict] = None, top_k: int = 10, n_neighbors: int = 6,
                session: Optional[Session] = None, dag_threshold: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Return {"suggestions": [...], "neighbors": [{image, label, similarity, source}], "health", "state"}."""
        t0 = time.perf_counter()
        kn = self.knowledge
        if kn is None:
            raise RuntimeError("engine has no knowledge base")
        pil = self._to_pil(image)
        z = np.asarray(self.encoder.encode_images([pil])[0], dtype=np.float32)
        meta_error = None
        try:
            meta_emb = self._meta_embedding(metadata)
        except Exception as exc:
            logger.warning("metadata embedding failed: %r", exc, exc_info=True)
            meta_emb, meta_error = None, repr(exc)
        ranked = rank_categories(
            z, meta_emb, kn, k=n_neighbors, top_k=top_k, dag_threshold=dag_threshold,
            overlay=session.overlay if session else None,
            head=session.head if session is not None and session.head_trained else None)
        suggestions, neighbours, health = ranked["suggestions"], ranked["neighbours"], ranked["health"]
        if meta_error:
            health["meta"] = {"status": "error", "detail": meta_error}
        for n in neighbours:
            if isinstance(n["image"], np.ndarray):
                n["image"] = Image.fromarray(np.asarray(n["image"], dtype=np.uint8))
        return {"suggestions": suggestions, "neighbors": neighbours, "health": health,
                "state": self.state_summary(), "latency": time.perf_counter() - t0}

    # ------------------------------------------------------------------ continual learning (per session)
    def _embed_cached(self, session: Session, pil: Image.Image) -> Tuple[str, np.ndarray]:
        key = hashlib.sha1(pil.tobytes()).hexdigest()
        if key not in session._emb_cache:
            session._emb_cache = {key: np.asarray(self.encoder.encode_images([pil])[0], dtype=np.float32)}
        z = session._emb_cache[key]
        return image_id_from_embedding(z), z

    def record_feedback(self, image_input: Any, category: str, accepted: bool, session: Optional[Session] = None) -> bool:
        """Store one accept/reject in THIS visitor's session. Accept also appends the image to the session
        overlay index (kNN sees it immediately, no retraining). Later clicks override earlier ones."""
        if session is None:
            raise ValueError("record_feedback needs the visitor's Session (engine state is read-only and shared)")
        kn = self.knowledge
        if kn is None or category not in set(kn.classes):
            return False
        from extract_embeddings import make_thumb
        pil = self._to_pil(image_input)
        iid, z = self._embed_cached(session, pil)
        session.feedback.record(iid, z, category, accepted)
        if accepted:
            session.overlay.accept(iid, z, category, thumb=make_thumb(pil))
        else:
            session.overlay.reject(iid, category)
        if session.log_path:
            try:
                append_event(session.log_path, iid, z, category, accepted)
            except OSError as exc:                       # e.g. read-only filesystem: feedback still works in memory
                logger.warning("could not append to feedback log %s: %s", session.log_path, exc)
        logger.info("feedback: %s %s (session %s, %s)", "accept" if accepted else "reject", category, session.id, session.counts())
        return True

    def train_supervised_head(self, epochs: int = 5, lr: float = 1e-4, batch_size: int = 32,
                              session: Optional[Session] = None, replay_rows: int = 256, holdout_frac: float = 0.2,
                              seed: int = 0) -> Dict[str, Any]:
        """Fine-tune a COPY of the head on this session's feedback with real mini-batches, a replay sample of the
        indexed training data, and ASL with three-state targets. Needs torch (imported here, lazily)."""
        if session is None:
            raise ValueError("train_supervised_head needs the visitor's Session")
        kn = self.knowledge
        X, T = session.feedback.to_training_arrays({c: i for i, c in enumerate(kn.classes)})
        if len(X) == 0:
            return {"loss_history": [0.0], "f1_score": 0.0, "f1_held_out": False, "samples": 0, "final_loss": 0.0}
        import torch
        from common import AsymmetricLoss
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        tr_idx, ho_idx = holdout_split(len(X), holdout_frac, seed)
        Xtr, Ttr = X[tr_idx], T[tr_idx]

        if session.head is not None:
            W0, b0 = session.head
        elif kn.head is not None:
            W0, b0 = kn.head
        else:                                                    # text-initialised classifier (PDF section 3)
            W0, b0 = kn.prototypes.copy(), np.zeros(len(kn.classes), np.float32)
        W = torch.nn.Parameter(torch.tensor(W0, dtype=torch.float32))
        b = torch.nn.Parameter(torch.tensor(b0, dtype=torch.float32))
        opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=1e-4)
        crit = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, margin=0.05)

        loss_history, batches_per_epoch, replay_used = [], 0, 0
        for _ in range(int(epochs)):
            Xe, Te = Xtr, Ttr
            rX, rT = replay_sample(*kn.replay_pool(max(replay_rows * 4, replay_rows), rng), n=replay_rows, rng=rng)
            if rX is not None:
                Xe, Te, replay_used = np.concatenate([Xtr, rX]), np.concatenate([Ttr, rT]), len(rX)
            total, n_batches = 0.0, 0
            for xb, tb in iterate_minibatches(Xe, Te, batch_size, rng):
                opt.zero_grad()
                loss = crit(torch.from_numpy(xb) @ W.T + b, torch.from_numpy(tb))
                loss.backward()
                opt.step()
                total += float(loss.item()); n_batches += 1
            batches_per_epoch = n_batches
            loss_history.append(total / max(n_batches, 1))

        Wn, bn = W.detach().numpy().astype("float32"), b.detach().numpy().astype("float32")
        held_out = len(ho_idx) > 0
        Xe_, Te_ = (X[ho_idx], T[ho_idx]) if held_out else (X, T)
        f1 = f1_labelled(1.0 / (1.0 + np.exp(-(Xe_ @ Wn.T + bn))), Te_)
        session.head, session.head_trained = (Wn, bn), True
        logger.info("head trained: loss %.4f, F1 %.3f (%s)", loss_history[-1], f1, "held-out" if held_out else "in-sample")
        return {"loss_history": loss_history, "f1_score": float(f1), "f1_held_out": held_out, "samples": int(len(X)),
                "final_loss": loss_history[-1], "batches_per_epoch": batches_per_epoch, "replay_rows": replay_used,
                "train_images": int(len(tr_idx)), "held_out_images": int(len(ho_idx))}


# =====================================================================
# SELF-TEST (no model download: uses the deterministic FakeEncoder)
# =====================================================================
if __name__ == "__main__":
    from encoders import FakeEncoder
    eng = CategoryEngine(encoder=FakeEncoder(), cache_dir=".cache/selftest")
    img = Image.new("RGB", (64, 64), (0, 100, 200))
    out = eng.suggest(img, metadata={"title": "A beautiful beach"}, top_k=3)
    assert out["suggestions"] and out["state"]["mode"] == "demo", out["state"]
    assert out["health"]["knn"]["status"] == "unavailable"          # demo KB has no index: honest, not a dummy vector
    sess = eng.new_session()
    assert eng.record_feedback(img, "Beaches", True, session=sess) and eng.record_feedback(img, "Mountains", False, session=sess)
    again = eng.suggest(img, top_k=3, session=sess)
    assert again["neighbors"] and again["neighbors"][0]["source"] == "session"
    print("ENGINE SELF-TEST PASSED (demo KB, fake encoder):", [s["category"] for s in out["suggestions"]])
