"""Bug 9: feedback per image {embedding, positives, negatives}, later clicks override; overlay index; real mini-batches; F1."""
import json

import numpy as np
import pytest

from feedback import (NEG, POS, UNK, FeedbackStore, SessionOverlay, append_event, f1_labelled, holdout_split,
                      image_id_from_embedding, iterate_minibatches, load_events, replay_sample)

CLASSES = ["A", "B", "C", "D"]
C2I = {c: i for i, c in enumerate(CLASSES)}


def emb(i, d=8):
    v = np.zeros(d, dtype="float32"); v[i % d] = 1.0
    return v


def test_later_click_overrides_earlier_click_for_same_image():
    s = FeedbackStore()
    e = emb(0); iid = image_id_from_embedding(e)
    s.record(iid, e, "A", accepted=True)
    s.record(iid, e, "A", accepted=False)           # changed their mind
    s.record(iid, e, "B", accepted=False)
    s.record(iid, e, "B", accepted=True)
    rec = s.records[iid]
    assert rec.positives == {"B"} and rec.negatives == {"A"}


def test_training_arrays_are_one_row_per_image_with_three_states():
    s = FeedbackStore()
    for i, (cat, acc) in enumerate([("A", True), ("B", False), ("C", True)]):
        e = emb(0); s.record(image_id_from_embedding(e), e, cat, acc)      # same image, three clicks
    e2 = emb(1); s.record(image_id_from_embedding(e2), e2, "D", False)
    X, T = s.to_training_arrays(C2I)
    assert X.shape == (2, 8) and T.shape == (2, 4)                          # NOT one row per click
    row = T[[i for i in range(2) if X[i, 0] == 1][0]]
    assert list(row) == [POS, NEG, POS, UNK]                                # reject is -1, never an all-zero 'unknown' row
    assert T[X[:, 1] == 1][0].tolist() == [UNK, UNK, UNK, NEG]


def test_unknown_categories_are_ignored_not_crashing():
    s = FeedbackStore()
    e = emb(0); s.record(image_id_from_embedding(e), e, "Not a class", True)
    X, T = s.to_training_arrays(C2I)
    assert (T == UNK).all()


def test_jsonl_is_append_only_and_replay_reproduces_state(tmp_path):
    path = tmp_path / "feedback" / "session.jsonl"
    s = FeedbackStore()
    e = emb(0); iid = image_id_from_embedding(e)
    for cat, acc in [("A", True), ("A", False), ("B", True)]:
        s.record(iid, e, cat, acc)
        append_event(path, iid, e, cat, acc)
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    size_before = path.stat().st_size
    append_event(path, iid, e, "C", True)
    assert path.stat().st_size > size_before and path.read_text().splitlines()[:3] == lines     # earlier lines untouched
    path.write_text(path.read_text() + '{"truncated": ')                   # crash mid-write
    replay, skipped = load_events(path)
    assert skipped == 1
    assert replay.records[iid].positives == {"B", "C"} and replay.records[iid].negatives == {"A"}
    assert json.loads(lines[0])["category"] == "A"


def test_overlay_accept_is_retrievable_without_retraining_and_reject_removes_label():
    ov = SessionOverlay(dim=8)
    e = emb(3); iid = image_id_from_embedding(e)
    ov.accept(iid, e, "B")
    D, I, labels = ov.search(e[None, :], k=3)
    assert I[0, 0] == 0 and labels(I[0, 0]) == ["B"]
    ov.accept(iid, e, "C")                                                  # same image: labels extend, no new vector
    assert ov.ntotal == 1 and sorted(ov.labels_of(iid)) == ["B", "C"]
    ov.reject(iid, "B")
    assert ov.labels_of(iid) == ["C"]
    ov.reject(iid, "C")
    assert ov.labels_of(iid) == [] and ov.search(e[None, :], 1)[2](0) is None      # empty-label rows must not vote


def test_minibatches_respect_batch_size_and_cover_everything_once():
    X = np.arange(25 * 3, dtype="float32").reshape(25, 3); T = np.zeros((25, 2), "float32")
    seen = []
    sizes = []
    for xb, tb in iterate_minibatches(X, T, batch_size=8, rng=np.random.default_rng(0)):
        sizes.append(len(xb)); seen += xb[:, 0].tolist()
    assert sizes == [8, 8, 8, 1] and sorted(seen) == sorted(X[:, 0].tolist())
    order2 = [xb[0, 0] for xb, _ in iterate_minibatches(X, T, 8, np.random.default_rng(1))]
    order1 = [xb[0, 0] for xb, _ in iterate_minibatches(X, T, 8, np.random.default_rng(0))]
    assert order1 != order2                                                # shuffled


def test_replay_sample_mixes_offline_rows_and_handles_small_pools():
    off_X = np.ones((100, 3), "float32"); off_T = np.zeros((100, 2), "float32")
    X, T = replay_sample(off_X, off_T, n=30, rng=np.random.default_rng(0))
    assert X.shape == (30, 3)
    X, T = replay_sample(off_X[:5], off_T[:5], n=30, rng=np.random.default_rng(0))
    assert len(X) == 5
    X, T = replay_sample(None, None, n=30, rng=np.random.default_rng(0))
    assert X is None and T is None


def test_f1_counts_only_labelled_entries():
    T = np.array([[POS, NEG, UNK, UNK]], dtype="float32")
    p = np.array([[0.9, 0.9, 0.9, 0.1]])             # TP=1 (A), FP=1 (B reject predicted), unknown entries ignored
    assert f1_labelled(p, T) == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert f1_labelled(np.array([[0.1, 0.1, 0.9, 0.9]]), T) == 0.0
    assert f1_labelled(p, np.zeros((1, 4), "float32")) == 0.0                # nothing labelled


def test_holdout_is_by_image_and_falls_back_when_tiny():
    tr, ho = holdout_split(20, frac=0.25, seed=0)
    assert set(tr).isdisjoint(ho) and len(ho) == 5 and len(tr) == 15
    tr, ho = holdout_split(3, frac=0.25, seed=0)
    assert len(ho) == 0 and len(tr) == 3                                   # too small: caller reports in-sample F1
