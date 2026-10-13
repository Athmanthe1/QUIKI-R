"""Bug 8: softmax kNN vote, SigLIP logit scale/bias for text, logistic stacking with missing flags, no min-max scaling."""
import numpy as np
import pytest

import fusion
from fusion import (FEATURES, Stacker, build_candidates, feature_matrix, fuse, fit_stacker, knn_vote, siglip_text_scores)


# ---------------- kNN ----------------
def test_knn_vote_is_bounded_and_does_not_saturate():
    sims = np.array([0.9] * 10)
    old = {"A": sum(sims)}                                          # the old scorer: a sum of cosines
    assert old["A"] > 8                                             # saturates / unbounded
    out = knn_vote(sims, [["A"]] * 10, tau=0.05)
    assert out["A"] == pytest.approx(1.0)                           # every neighbour agrees -> exactly 1
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_knn_vote_weights_closer_neighbours_more():
    sims = np.array([0.95, 0.60, 0.60])
    out = knn_vote(sims, [["Near"], ["Far1"], ["Far2"]], tau=0.05)
    assert out["Near"] > 0.9 and out["Near"] > out["Far1"] + out["Far2"]


def test_knn_vote_multilabel_and_empty_and_padding():
    out = knn_vote(np.array([0.8, 0.8]), [["A", "B"], ["A"]], tau=0.05)
    assert out["A"] == pytest.approx(1.0) and out["B"] == pytest.approx(0.5)
    assert knn_vote(np.array([]), [], tau=0.05) == {}
    assert knn_vote(np.array([0.9, -np.inf]), [["A"], None], tau=0.05)["A"] == pytest.approx(1.0)   # faiss padding rows


# ---------------- text scorer ----------------
def test_siglip_text_scores_use_logit_scale_and_bias():
    z = np.array([1.0, 0.0]); T = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    ls, lb = np.log(10.0), -5.0
    got = siglip_text_scores(z, T, ls, lb)
    expect = 1 / (1 + np.exp(-(10.0 * (T @ z) - 5.0)))
    assert np.allclose(got, expect)
    assert got[0] > got[1] > got[2] and ((0 < got) & (got < 1)).all()


# ---------------- candidates and features ----------------
def test_candidates_are_union_of_per_scorer_topn_and_skip_missing_scorers():
    scores = {"knn": {"A": 0.9, "B": 0.1, "C": 0.05}, "txt": {"B": 0.8, "D": 0.7, "E": 0.1}, "cls": None, "meta": None}
    assert set(build_candidates(scores, top_n=2)) == {"A", "B", "D"}
    assert set(build_candidates(scores, top_n=200)) == {"A", "B", "C", "D", "E"}


def test_feature_matrix_has_missing_flags():
    scores = {"knn": {"A": 0.9}, "txt": {"A": 0.2, "B": 0.4}, "cls": None, "meta": None}
    X = feature_matrix(["A", "B"], scores)
    assert X.shape == (2, 2 * len(FEATURES))
    f = {name: i for i, name in enumerate(fusion.FEATURE_NAMES)}
    assert X[0, f["knn"]] == 0.9 and X[1, f["knn"]] == 0.0 and X[0, f["miss_knn"]] == 0.0   # present, no evidence for B
    assert (X[:, f["miss_cls"]] == 1).all() and (X[:, f["cls"]] == 0).all()
    assert (X[:, f["miss_meta"]] == 1).all()


# ---------------- stacking ----------------
def synthetic_examples(n_images=300, seed=0, drop=()):
    rng = np.random.default_rng(seed)
    cats = [f"c{i}" for i in range(30)]
    ex = []
    for _ in range(n_images):
        truth = set(rng.choice(cats, size=2, replace=False))
        s = {"knn": {}, "txt": {}, "cls": {}, "meta": {}}
        for c in cats:
            y = c in truth
            s["knn"][c] = float(np.clip(rng.normal(0.8 if y else 0.1, 0.1), 0, 1))     # informative
            s["txt"][c] = float(np.clip(rng.normal(0.55 if y else 0.45, 0.2), 0, 1))   # weakly informative
            s["cls"][c] = float(np.clip(rng.normal(0.6 if y else 0.3, 0.2), 0, 1))
            s["meta"][c] = float(rng.uniform(0, 1))                                    # pure noise
        for d in drop:
            s[d] = None
        ex.append((s, truth))
    return ex


def test_stacker_learns_weights_and_is_calibrated():
    train, test = synthetic_examples(300, 0), synthetic_examples(100, 1)
    st = fit_stacker(train, top_n=200)
    w = dict(zip(fusion.FEATURE_NAMES, st.coef))
    assert w["knn"] > w["txt"] > abs(w["meta"]) - 0.5                # informative scorer gets the biggest weight
    assert abs(w["meta"]) < w["knn"] / 3
    preds, labels = [], []
    for s, truth in test:
        cands = build_candidates(s, 200)
        p = st.predict_proba(feature_matrix(cands, s))
        preds += list(p); labels += [c in truth for c in cands]
    preds, labels = np.array(preds), np.array(labels, float)
    assert abs(preds.mean() - labels.mean()) < 0.03                  # calibration-in-the-large
    hi = preds > 0.8
    assert hi.sum() > 10 and labels[hi].mean() > 0.75                # high-confidence bucket really is mostly right


def test_missing_scorer_gets_flag_not_weight():
    st = fit_stacker(synthetic_examples(200, 2, drop=("cls",)), top_n=200)
    w = dict(zip(fusion.FEATURE_NAMES, st.coef))
    assert abs(w["cls"]) < 1e-6                                       # constant-zero feature carries no weight
    s, truth = synthetic_examples(1, 5, drop=("cls",))[0]
    out = fuse(s, st, top_n=200)
    assert out and all("cls" not in o["scores"] for o in out) and out[0]["missing"] == ["cls"]


def test_fuse_returns_probabilities_not_minmax_scaled():
    st = fit_stacker(synthetic_examples(200, 3), top_n=200)
    s, truth = synthetic_examples(1, 9)[0]
    # make the query genuinely ambiguous: all scorers near their "negative" level
    s = {k: {c: 0.3 for c in v} for k, v in s.items()}
    out = fuse(s, st, top_n=200)
    assert out[0]["confidence"] < 0.5                                 # old min-max scaling forced 0.40-0.98 bands
    assert out[0]["calibrated"] is True
    confs = [o["confidence"] for o in out]
    assert confs == sorted(confs, reverse=True)


def test_uncalibrated_fallback_is_flagged_and_ignores_missing_scorers():
    s = {"knn": None, "txt": {"A": 0.2, "B": 0.1}, "cls": None, "meta": None}
    out = fuse(s, None, top_n=200)
    assert out[0]["category"] == "A" and out[0]["confidence"] == pytest.approx(0.2)   # mean of PRESENT scorers only
    assert out[0]["calibrated"] is False and set(out[0]["missing"]) == {"knn", "cls", "meta"}


def test_stacker_json_roundtrip(tmp_path):
    st = fit_stacker(synthetic_examples(100, 4), top_n=200)
    st.save(tmp_path / "stacker.json")
    st2 = Stacker.load(tmp_path / "stacker.json")
    X = np.random.default_rng(0).uniform(size=(5, len(fusion.FEATURE_NAMES)))
    assert np.allclose(st.predict_proba(X), st2.predict_proba(X))


def test_fit_needs_both_classes():
    with pytest.raises(ValueError, match="positive"):
        fit_stacker([({"knn": {"A": 0.5}, "txt": None, "cls": None, "meta": None}, set())], top_n=10)


def _informative_meta_examples(n, seed, hide_meta):
    rng = np.random.default_rng(seed)
    cats = [f"c{i}" for i in range(30)]
    out = []
    for _ in range(n):
        truth = set(rng.choice(cats, size=2, replace=False))
        s = {"knn": {}, "txt": {}, "cls": None, "meta": {}}
        for c in cats:
            y = c in truth
            s["knn"][c] = float(np.clip(rng.normal(0.6 if y else 0.3, 0.2), 0, 1))
            s["txt"][c] = float(np.clip(rng.normal(0.55 if y else 0.45, 0.2), 0, 1))
            s["meta"][c] = float(np.clip(rng.normal(0.9 if y else 0.1, 0.05), 0, 1))     # very informative
        if hide_meta:
            s["meta"] = None
        out.append((s, truth))
    return out


def test_missing_flags_keep_probabilities_calibrated_when_a_scorer_is_hidden_at_serving_time():
    train = _informative_meta_examples(400, 0, hide_meta=False)
    plain = fit_stacker(train, top_n=200)
    aug = fit_stacker(train, top_n=200, missing_augment={"meta": 0.5})
    test = _informative_meta_examples(150, 1, hide_meta=True)           # users upload without a description

    def gap(st):
        preds, labels = [], []
        for s, truth in test:
            cands = build_candidates(s, 200)
            preds += list(st.predict_proba(feature_matrix(cands, s)))
            labels += [c in truth for c in cands]
        return abs(np.mean(preds) - np.mean(labels))

    assert gap(aug) < 0.03
    assert gap(aug) < gap(plain)             # without augmentation the model is over-confident when meta is absent
