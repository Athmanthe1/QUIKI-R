"""scoring.py - the four scorers + neighbours, from embeddings only (torch-free).

The ONE implementation of "embeddings -> per-scorer scores" shared by the engine (serving),
fit_stacker.py (fitting the fusion) and evaluate.py (measuring it), so the stages cannot drift apart.
Scorers never fail silently: an exception is logged and surfaced in ``health``; an unavailable
scorer is reported as such and contributes a missing flag in fusion, never weight.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from feedback import SessionOverlay
from fusion import DEFAULT_TAU, Scores, Stacker, knn_vote, siglip_text_scores

logger = logging.getLogger("CommonsScoring")


@dataclass
class Knowledge:
    """Read-only knowledge base. Built once; never mutated by serving, feedback or training."""
    classes: List[str]
    prototypes: Optional[np.ndarray] = None            # [C, D] L2-normalised text prototypes
    index: Any = None                                  # faiss / NumpyFlatIndex over TRAIN images, or None
    index_labels: List[List[str]] = field(default_factory=list)   # class NAMES per index row
    thumbs: Optional[np.ndarray] = None                # uint8 [N, 64, 64, 3], row-aligned with the index
    head: Optional[Tuple[np.ndarray, np.ndarray]] = None          # offline head (W [C, D], b [C])
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    stacker: Optional[Stacker] = None
    logit_scale: float = float(np.log(10.0))
    logit_bias: float = -5.0
    source: str = "demo"                               # "artifacts" | "demo"
    index_type: str = "none"
    notes: List[str] = field(default_factory=list)

    def replay_pool(self, max_rows: int, rng: np.random.Generator) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Sample indexed train embeddings + their labels (as +1 / unknown targets) for replay during fine-tuning."""
        if self.index is None or not getattr(self.index, "ntotal", 0) or not self.index_labels:
            return None, None
        n = self.index.ntotal
        rows = np.sort(rng.choice(n, size=min(max_rows, n), replace=False))
        try:
            X = np.stack([self.index.reconstruct(int(i)) for i in rows]).astype(np.float32)
        except Exception as exc:                       # HNSW / IVF-PQ may not support reconstruct
            logger.info("replay pool unavailable (%s)", exc)
            return None, None
        c2i = {c: i for i, c in enumerate(self.classes)}
        T = np.zeros((len(rows), len(self.classes)), np.float32)
        for k, r in enumerate(rows):
            for c in self.index_labels[r]:
                if c in c2i:
                    T[k, c2i[c]] = 1.0
        return X, T


def _health_ok(detail: str = "") -> Dict[str, str]:
    return {"status": "ok", "detail": detail}


def compute_scores(z: np.ndarray, meta_emb: Optional[np.ndarray], kn: Knowledge, *,
                   overlay: Optional[SessionOverlay] = None,
                   head: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                   k: int = 10, tau: float = DEFAULT_TAU) -> Tuple[Scores, List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    """Return (scores, neighbours, health). ``head`` overrides the offline head (session-trained copy)."""
    scores: Scores = {"knn": None, "cls": None, "txt": None, "meta": None}
    health = {n: {"status": "unavailable", "detail": ""} for n in scores}
    neighbours: List[Dict[str, Any]] = []
    classes = kn.classes

    # ---- kNN: main index + this session's accepted images ----
    try:
        cands: List[Tuple[float, Optional[List[str]], Any, str]] = []
        if kn.index is not None and kn.index.ntotal > 0:
            D, I = kn.index.search(z.reshape(1, -1).astype(np.float32), min(k, kn.index.ntotal))
            for sim, row in zip(D[0], I[0]):
                if row < 0 or row >= len(kn.index_labels):
                    continue
                thumb = kn.thumbs[row] if kn.thumbs is not None and row < len(kn.thumbs) else None
                cands.append((float(sim), kn.index_labels[row], thumb, "index"))
        if overlay is not None and overlay.ntotal > 0:
            D, I, lookup = overlay.search(z.reshape(1, -1).astype(np.float32), min(k, overlay.ntotal))
            for sim, row in zip(D[0], I[0]):
                labs = lookup(int(row))
                if row >= 0 and labs:
                    cands.append((float(sim), labs, overlay.thumb_of_row(int(row)), "session"))
        cands.sort(key=lambda c: -c[0])
        cands = cands[:k]
        if cands:
            scores["knn"] = knn_vote(np.array([c[0] for c in cands]), [c[1] for c in cands], tau)
            neighbours = [{"image": c[2], "label": ", ".join(c[1][:3]), "similarity": c[0], "source": c[3]} for c in cands]
            health["knn"] = _health_ok(f"{len(cands)} neighbours")
        else:
            health["knn"]["detail"] = "no index loaded" if kn.index is None else "empty index"
    except Exception as exc:
        logger.warning("kNN scorer failed: %r", exc, exc_info=True)
        scores["knn"], health["knn"] = None, {"status": "error", "detail": repr(exc)}

    # ---- zero-shot text prototypes ----
    try:
        if kn.prototypes is not None and len(kn.prototypes):
            p = siglip_text_scores(z, kn.prototypes, kn.logit_scale, kn.logit_bias)
            scores["txt"] = {c: float(p[i]) for i, c in enumerate(classes)}
            health["txt"] = _health_ok(f"{len(classes)} prototypes")
        else:
            health["txt"]["detail"] = "no text prototypes"
    except Exception as exc:
        logger.warning("text scorer failed: %r", exc, exc_info=True)
        scores["txt"], health["txt"] = None, {"status": "error", "detail": repr(exc)}

    # ---- metadata text (same SigLIP head as the text scorer) ----
    try:
        if meta_emb is not None and kn.prototypes is not None and len(kn.prototypes):
            p = siglip_text_scores(meta_emb, kn.prototypes, kn.logit_scale, kn.logit_bias)
            scores["meta"] = {c: float(p[i]) for i, c in enumerate(classes)}
            health["meta"] = _health_ok("title/description scored")
        else:
            health["meta"]["detail"] = "no title/description given" if meta_emb is None else "no text prototypes"
    except Exception as exc:
        logger.warning("metadata scorer failed: %r", exc, exc_info=True)
        scores["meta"], health["meta"] = None, {"status": "error", "detail": repr(exc)}

    # ---- supervised head (offline, or the session-trained copy) ----
    try:
        h = head if head is not None else kn.head
        if h is not None:
            W, b = h
            p = 1.0 / (1.0 + np.exp(-(W @ z + b)))
            scores["cls"] = {c: float(p[i]) for i, c in enumerate(classes)}
            health["cls"] = _health_ok("session-trained head" if head is not None else "offline head")
        else:
            health["cls"]["detail"] = "no trained head"
    except Exception as exc:
        logger.warning("classifier scorer failed: %r", exc, exc_info=True)
        scores["cls"], health["cls"] = None, {"status": "error", "detail": repr(exc)}

    return scores, neighbours, health
