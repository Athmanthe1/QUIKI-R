"""Independent numpy oracles written directly from the problem-statement formulas (PDF section 3).

They are used to check the torch implementations when torch is available, and on their own
to pin down what the spec says (and where the old code disagreed with it).
"""
import numpy as np

POS, UNK, NEG = 1, 0, -1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def asl_spec(logits, targets, gamma_neg=4.0, gamma_pos=0.0, margin=0.05, unknown_weight=1.0, eps=1e-8):
    """Per-sample sum over classes, mean over batch.

    POS : y=1                      -> -(1-p)^g+ * log p
    UNK : unlabelled (tolerated)   -> -p_m^g-  * log(1-p_m),  p_m = max(p-m, 0)
    NEG : explicit reject          -> -log(1-p)     (full strength: no margin, no focal weight)
    """
    p = sigmoid(logits)
    t = np.asarray(targets)
    pm = np.maximum(p - margin, 0.0)
    pos = -((1 - p) ** gamma_pos) * np.log(p + eps)
    unk = -(pm ** gamma_neg) * np.log(np.clip(1 - pm, eps, None))
    neg = -np.log(np.clip(1 - p, eps, None))
    loss = np.where(t == POS, pos, np.where(t == NEG, neg, unknown_weight * unk))
    return float(loss.sum(axis=-1).mean())


def asl_old(logits, targets, gamma_neg=4.0, gamma_pos=1.0, margin=0.05, eps=1e-8):
    """The formula the repo shipped: log(1 - p) (NOT log(1 - p_m)) in the negative term."""
    p = sigmoid(logits)
    y = (np.asarray(targets) == POS).astype(np.float64)
    pm = np.maximum(p - margin, 0.0)
    return float((-(y * (1 - p) ** gamma_pos * np.log(p + eps)
                    + (1 - y) * pm ** gamma_neg * np.log(1 - p + eps))).sum(axis=-1).mean())
