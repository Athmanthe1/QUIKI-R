"""Metrics (PDF section 5), checked against hand-computed values."""
import math

import networkx as nx
import numpy as np
import pytest

import metrics as M


def test_prf_at_k_hand_computed():
    pred = ["A", "B", "C", "D", "E"]; truth = {"A", "C", "X"}
    p, r, f = M.prf_at_k(pred, truth, 3)                       # hits in top3: A, C
    assert p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3) and f == pytest.approx(2 / 3)
    p, r, f = M.prf_at_k(pred, truth, 1)
    assert (p, r) == (1.0, pytest.approx(1 / 3)) and f == pytest.approx(2 * 1 * (1 / 3) / (1 + 1 / 3))
    assert M.prf_at_k(["Z"], truth, 5) == (0.0, 0.0, 0.0)
    assert M.prf_at_k(pred, set(), 3) == (0.0, 0.0, 0.0)       # nothing to find
    p, _, _ = M.prf_at_k(["A"], {"A"}, 5)
    assert p == pytest.approx(1 / 5)                           # P@k divides by k even if fewer are returned


def test_sample_wise_average_skips_nothing_and_averages_per_image():
    out = M.sample_wise([["A", "B"], ["Z", "Y"]], [{"A"}, {"Q"}], ks=(1, 2))
    assert out[1]["P"] == pytest.approx(0.5) and out[2]["P"] == pytest.approx(0.25) and out[2]["R"] == pytest.approx(0.5)


def test_average_precision():
    # ranked by score: pos, neg, pos -> AP = (1/1 + 2/3) / 2
    assert M.average_precision(np.array([0.9, 0.8, 0.7]), np.array([1, 0, 1])) == pytest.approx((1 + 2 / 3) / 2)
    assert M.average_precision(np.array([0.1, 0.2]), np.array([0, 0])) is None      # no positives -> undefined
    assert M.average_precision(np.array([0.9, 0.1]), np.array([1, 0])) == 1.0


def test_map_over_categories_skips_categories_without_positives():
    S = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.5], [0.7, 0.3, 0.5]])
    Y = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    m, per = M.mean_average_precision(S, Y)
    assert m == pytest.approx(1.0) and per[2] is None and len(per) == 3


def test_macro_over_categories():
    cats = ["A", "B"]
    preds = [["A", "B"], ["A", "B"], ["B", "A"]]
    truths = [{"A"}, {"A"}, {"B"}]
    out = M.macro_at_k(preds, truths, cats, k=1)
    # A: predicted at top1 for imgs 0,1 (both true) -> P=1, R=2/2=1 ; B: predicted top1 for img2 (true) -> P=1,R=1
    assert out["P"] == pytest.approx(1.0) and out["R"] == pytest.approx(1.0) and out["n_categories"] == 2
    out2 = M.macro_at_k([["B"], ["B"], ["B"]], truths, cats, k=1)
    assert out2["R"] == pytest.approx((0 + 1) / 2)             # A never found, B found


def test_hierarchy_aware_precision_gives_partial_credit_to_ancestors_and_descendants():
    g = nx.DiGraph([("Nature", "Mountains"), ("Mountains", "Alps")])
    truth = {"Mountains"}
    assert M.hier_precision_at_k(["Mountains"], truth, g, 1) == 1.0
    assert M.hier_precision_at_k(["Nature"], truth, g, 1) == 0.5        # ancestor
    assert M.hier_precision_at_k(["Alps"], truth, g, 1) == 0.5          # descendant
    assert M.hier_precision_at_k(["Cats"], truth, g, 1) == 0.0
    assert M.hier_precision_at_k(["Nature", "Cats"], truth, g, 2) == 0.25


def test_frequency_buckets_are_terciles_by_train_count():
    counts = {"a": 100, "b": 90, "c": 50, "d": 40, "e": 5, "f": 3}
    b = M.frequency_buckets(counts)
    assert b["head"] == {"a", "b"} and b["torso"] == {"c", "d"} and b["tail"] == {"e", "f"}


def test_resolution_bands():
    assert M.resolution_band(100, 5000) == "low (<256px)"             # shorter side decides
    assert M.resolution_band(800, 600) == "mid (256-1023px)"
    assert M.resolution_band(2000, 1500) == "high (>=1024px)"
    assert M.resolution_band(0, 0) == "unknown"


def test_latency_summary():
    s = M.latency_summary([0.010] * 90 + [0.100] * 10)
    assert s["p50_ms"] == pytest.approx(10.0) and s["p95_ms"] == pytest.approx(100.0) and s["throughput_per_s"] > 0


def test_wilson_interval():
    lo, hi = M.wilson_interval(8, 10)
    assert 0.45 < lo < 0.55 and 0.9 < hi < 0.98
    assert M.wilson_interval(0, 0) == (0.0, 1.0)


def test_reviewed_precision_extrapolates_from_a_sample_of_wrong_suggestions():
    # strict P@k = 0.40 over 100 suggestions -> 60 wrong. 20 sampled, 10 judged correct (unsure ignored)
    verdicts = ["correct"] * 10 + ["incorrect"] * 8 + ["unsure"] * 2
    out = M.reviewed_precision(strict_precision=0.40, n_suggestions=100, verdicts=verdicts)
    assert out["fraction_wrong_actually_correct"] == pytest.approx(10 / 18)
    assert out["reviewed_precision"] == pytest.approx(0.40 + 0.60 * 10 / 18)
    assert out["n_reviewed"] == 18 and out["ci95"][0] < out["fraction_wrong_actually_correct"] < out["ci95"][1]
    assert M.reviewed_precision(0.4, 100, [])["reviewed_precision"] is None
