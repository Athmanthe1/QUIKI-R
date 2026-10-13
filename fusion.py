"""fusion.py - scorer maths and calibrated stacking (PDF section 3). numpy + scikit-learn only.

* kNN       : softmax-weighted vote over the K neighbours, a value in [0, 1] per category.
* text/meta : SigLIP's own pairwise-sigmoid head, sigmoid(exp(logit_scale) * z.t + logit_bias).
* stacking  : logistic regression over [s_knn, s_cls, s_txt, s_meta] + one missing-flag per scorer,
              fit on the validation split over the UNION of each scorer's top-N candidates.
* A scorer that is not available contributes value 0 and flag 1, never weight.
Probabilities are reported as they are: no min-max rescaling.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

FEATURES: Tuple[str, ...] = ("knn", "cls", "txt", "meta")          # order of the PDF's w^T [s_knn, s_cls, s_txt, s_meta]
FEATURE_NAMES: List[str] = list(FEATURES) + [f"miss_{f}" for f in FEATURES]
DEFAULT_TOP_N = 200
DEFAULT_TAU = 0.05

Scores = Dict[str, Optional[Dict[str, float]]]                      # scorer -> {category: score}, or None if unavailable


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def knn_vote(sims: np.ndarray, neighbor_labels: Sequence[Optional[Sequence[str]]], tau: float = DEFAULT_TAU) -> Dict[str, float]:
    """s_knn(c|x) = sum_i exp(sim_i/tau) 1[c in Y_i] / sum_i exp(sim_i/tau), over valid neighbours."""
    sims = np.asarray(sims, dtype=np.float64)
    if sims.size == 0:
        return {}
    valid = np.isfinite(sims) & np.array([lab is not None for lab in neighbor_labels], dtype=bool)
    if not valid.any():
        return {}
    s = sims[valid]
    w = np.exp((s - s.max()) / tau)
    w /= w.sum()
    out: Dict[str, float] = {}
    for weight, labs in zip(w, [l for l, v in zip(neighbor_labels, valid) if v]):
        for c in set(labs):
            out[c] = out.get(c, 0.0) + float(weight)
    return {c: float(min(max(v, 0.0), 1.0)) for c, v in out.items()}


def siglip_text_scores(z: np.ndarray, prototypes: np.ndarray, logit_scale: float, logit_bias: float) -> np.ndarray:
    """Per-class probability from SigLIP's own head. ``logit_scale`` is the LOG scale parameter as stored in the model."""
    return _sigmoid(np.exp(logit_scale) * (prototypes @ z) + logit_bias)


def build_candidates(scores: Scores, top_n: int = DEFAULT_TOP_N) -> List[str]:
    """Union of each available scorer's top-N categories, ordered by best score then name."""
    best: Dict[str, float] = {}
    for name in FEATURES:
        s = scores.get(name)
        if not s:
            continue
        top = sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        for c, v in top:
            best[c] = max(best.get(c, -np.inf), v)
    return [c for c, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]


def feature_matrix(cands: Sequence[str], scores: Scores) -> np.ndarray:
    X = np.zeros((len(cands), 2 * len(FEATURES)), dtype=np.float64)
    for j, name in enumerate(FEATURES):
        s = scores.get(name)
        if s is None:
            X[:, len(FEATURES) + j] = 1.0                                   # missing flag; value stays 0
        else:
            X[:, j] = [s.get(c, 0.0) for c in cands]
    return X


class Stacker:
    """Logistic regression  p = sigmoid(w . features + b)  with JSON persistence."""

    def __init__(self, coef: Sequence[float], intercept: float, meta: Optional[Dict[str, Any]] = None):
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.meta = meta or {}
        if len(self.coef) != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} coefficients, got {len(self.coef)}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(X, dtype=np.float64) @ self.coef + self.intercept)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps({"feature_names": FEATURE_NAMES, "coef": self.coef.tolist(),
                                          "intercept": self.intercept, "meta": self.meta}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Stacker":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw["feature_names"] != FEATURE_NAMES:
            raise ValueError("stacker feature layout does not match this code version")
        return cls(raw["coef"], raw["intercept"], raw.get("meta"))


def fit_stacker(examples: Iterable[Tuple[Scores, Set[str]]], top_n: int = DEFAULT_TOP_N, C: float = 1.0,
                missing_augment: Optional[Dict[str, float]] = None, seed: int = 0) -> Stacker:
    """Fit on validation examples ``(scorer outputs, true category set)``.

    Rows are the union-of-top-N candidates; label = candidate in the true set. True categories that
    no scorer proposed are recall misses and cannot be learned from here. ``missing_augment``
    maps scorer -> probability of hiding that scorer for an image, so the missing flags are
    learned from data (e.g. {"meta": 0.3}: users often upload without a description).
    """
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for scores, truth in examples:
        scores = dict(scores)
        for name, prob in (missing_augment or {}).items():
            if scores.get(name) is not None and rng.random() < prob:
                scores[name] = None
        cands = build_candidates(scores, top_n)
        if not cands:
            continue
        Xs.append(feature_matrix(cands, scores))
        ys.extend(1 if c in truth else 0 for c in cands)
    if not Xs:
        raise ValueError("no candidates produced; cannot fit the stacker")
    X, y = np.vstack(Xs), np.array(ys)
    if y.sum() == 0:
        raise ValueError("no positive candidates in the validation data")
    if y.sum() == len(y):
        raise ValueError("no negative candidates in the validation data")
    lr = LogisticRegression(C=C, max_iter=2000).fit(X, y)
    return Stacker(lr.coef_[0], lr.intercept_[0],
                   {"n_rows": int(len(y)), "positive_rate": float(y.mean()), "top_n": top_n, "C": C})


def fuse(scores: Scores, stacker: Optional[Stacker], top_n: int = DEFAULT_TOP_N) -> List[Dict[str, Any]]:
    """Return suggestions sorted by probability. ``calibrated`` is False without a fitted stacker."""
    missing = [n for n in FEATURES if scores.get(n) is None]
    cands = build_candidates(scores, top_n)
    if not cands:
        return []
    if stacker is not None:
        conf = stacker.predict_proba(feature_matrix(cands, scores))
        calibrated = True
    else:
        present = [n for n in FEATURES if scores.get(n) is not None]
        conf = np.array([np.mean([scores[n].get(c, 0.0) for n in present]) for c in cands])
        calibrated = False
    out = []
    for c, p in zip(cands, conf):
        out.append({"category": c, "confidence": float(p), "calibrated": calibrated, "missing": list(missing),
                    "scores": {n: float(scores[n].get(c, 0.0)) for n in FEATURES if scores.get(n) is not None}})
    return sorted(out, key=lambda o: (-o["confidence"], o["category"]))
