"""metrics.py - evaluation metrics from the problem statement, section 5 (numpy only, torch-free)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np


def prf_at_k(pred: Sequence[str], truth: Set[str], k: int) -> Tuple[float, float, float]:
    """P@k = |top-k ∩ Y| / k,  R@k = |top-k ∩ Y| / |Y|,  F1@k = 2PR/(P+R). Empty Y -> zeros."""
    if not truth:
        return 0.0, 0.0, 0.0
    hits = len(set(pred[:k]) & truth)
    p, r = hits / k, hits / len(truth)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def sample_wise(preds: Sequence[Sequence[str]], truths: Sequence[Set[str]], ks=(1, 3, 5, 10)) -> Dict[int, Dict[str, float]]:
    """Average of per-image P/R/F1 (user-facing quality)."""
    out = {}
    for k in ks:
        vals = np.array([prf_at_k(p, t, k) for p, t in zip(preds, truths)]) if preds else np.zeros((0, 3))
        m = vals.mean(axis=0) if len(vals) else np.zeros(3)
        out[k] = {"P": float(m[0]), "R": float(m[1]), "F1": float(m[2]), "n": int(len(vals))}
    return out


def average_precision(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """AP of one category: rank images by score, average precision at each positive. None if no positives."""
    labels = np.asarray(labels).astype(bool)
    if not labels.any():
        return None
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    hit = labels[order]
    prec = np.cumsum(hit) / (np.arange(len(hit)) + 1)
    return float(prec[hit].mean())


def mean_average_precision(S: np.ndarray, Y: np.ndarray) -> Tuple[Optional[float], List[Optional[float]]]:
    """mAP over categories; S, Y are [n_images, n_categories]. Categories without positives are skipped."""
    per = [average_precision(S[:, c], Y[:, c]) for c in range(S.shape[1])]
    valid = [a for a in per if a is not None]
    return (float(np.mean(valid)) if valid else None), per


def macro_at_k(preds: Sequence[Sequence[str]], truths: Sequence[Set[str]], categories: Sequence[str], k: int) -> Dict[str, float]:
    """Per-category P/R/F1 at k, averaged over categories that occur in the ground truth (tail behaviour)."""
    top = [set(p[:k]) for p in preds]
    P, R, F = [], [], []
    for c in categories:
        n_true = sum(c in t for t in truths)
        if n_true == 0:
            continue
        n_pred = sum(c in p for p in top)
        tp = sum(c in p and c in t for p, t in zip(top, truths))
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_true
        P.append(p); R.append(r); F.append(2 * p * r / (p + r) if p + r else 0.0)
    n = len(P)
    return {"P": float(np.mean(P)) if n else 0.0, "R": float(np.mean(R)) if n else 0.0,
            "F1": float(np.mean(F)) if n else 0.0, "n_categories": n}


def hier_precision_at_k(pred: Sequence[str], truth: Set[str], graph: nx.DiGraph, k: int) -> float:
    """Hierarchy-aware precision: exact hit = 1, a predicted ancestor or descendant of a true label = 0.5."""
    total = 0.0
    for c in pred[:k]:
        if c in truth:
            total += 1.0
        elif c in graph and any(t in graph and (t in nx.descendants(graph, c) or t in nx.ancestors(graph, c)) for t in truth):
            total += 0.5
    return total / k


def frequency_buckets(train_counts: Dict[str, int]) -> Dict[str, Set[str]]:
    """Head / torso / tail = terciles of the class list ordered by train frequency (ties broken by name)."""
    ordered = sorted(train_counts, key=lambda c: (-train_counts[c], c))
    n = len(ordered)
    a, b = math.ceil(n / 3), math.ceil(2 * n / 3)
    return {"head": set(ordered[:a]), "torso": set(ordered[a:b]), "tail": set(ordered[b:])}


def resolution_band(width: int, height: int) -> str:
    side = min(int(width or 0), int(height or 0))
    if side <= 0:
        return "unknown"
    if side < 256:
        return "low (<256px)"
    if side < 1024:
        return "mid (256-1023px)"
    return "high (>=1024px)"


def latency_summary(seconds: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(seconds, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "p50_ms": float("nan"), "p95_ms": float("nan"), "mean_ms": float("nan"), "throughput_per_s": float("nan")}
    return {"n": int(a.size), "p50_ms": float(np.percentile(a, 50, method="nearest") * 1e3),
            "p95_ms": float(np.percentile(a, 95, method="nearest") * 1e3), "mean_ms": float(a.mean() * 1e3),
            "throughput_per_s": float(1.0 / a.mean())}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def reviewed_precision(strict_precision: float, n_suggestions: int, verdicts: Sequence[str]) -> Dict[str, object]:
    """Human-reviewed score. ``verdicts`` are reviewer judgements ('correct' / 'incorrect' / 'unsure') of a RANDOM
    sample of the suggestions that the stored labels call wrong. Stored labels are incomplete, so some of those are
    right. Estimate that fraction f (unsure ignored) and add it back:  reviewed = strict + (1 - strict) * f."""
    judged = [v for v in verdicts if v in ("correct", "incorrect")]
    if not judged:
        return {"reviewed_precision": None, "fraction_wrong_actually_correct": None, "n_reviewed": 0, "ci95": None}
    correct = sum(v == "correct" for v in judged)
    f = correct / len(judged)
    lo, hi = wilson_interval(correct, len(judged))
    return {"reviewed_precision": strict_precision + (1 - strict_precision) * f, "fraction_wrong_actually_correct": f,
            "n_reviewed": len(judged), "ci95": (lo, hi),
            "reviewed_precision_ci95": (strict_precision + (1 - strict_precision) * lo, strict_precision + (1 - strict_precision) * hi)}
