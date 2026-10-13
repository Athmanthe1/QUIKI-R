"""Bug 1: ASL negative term must be p_m^g * log(1 - p_m); targets are three-state (positive / unknown / hard negative)."""
import numpy as np
import pytest

from tests.reference import NEG, POS, UNK, asl_old, asl_spec, sigmoid

# ---------- pure numpy: pins the spec and shows the old formula disagrees (runs everywhere) ----------

def test_old_formula_disagrees_with_spec_on_unknown_negatives():
    logits = np.array([[2.0, 1.5, -1.0]])              # first two classes are confident false positives
    targets = np.array([[POS, UNK, UNK]])
    assert abs(asl_old(logits, targets, gamma_pos=0.0) - asl_spec(logits, targets)) > 1e-3


def test_spec_values_by_hand():
    x = 1.0
    p = float(sigmoid(x)); pm = p - 0.05
    expect_unk = -(pm ** 4) * np.log(1 - pm)
    assert asl_spec(np.array([[x]]), np.array([[UNK]])) == pytest.approx(expect_unk, rel=1e-6)
    assert asl_spec(np.array([[x]]), np.array([[NEG]])) == pytest.approx(-np.log(1 - p), rel=1e-6)
    assert asl_spec(np.array([[x]]), np.array([[POS]]), gamma_pos=1.0) == pytest.approx(-(1 - p) * np.log(p), rel=1e-6)


def test_margin_zeroes_loss_for_easy_unknowns_and_reject_is_stronger():
    easy = np.array([[-5.0]])                              # p ~ 0.0067 < margin
    assert asl_spec(easy, np.array([[UNK]])) == 0.0
    assert asl_spec(easy, np.array([[NEG]])) > 0.0         # an explicit reject still penalises
    mid = np.array([[0.5]])
    assert asl_spec(mid, np.array([[NEG]])) > asl_spec(mid, np.array([[UNK]]))


# ---------- torch implementation vs the oracle (skipped when torch is unavailable) ----------

@pytest.fixture(scope="module")
def torch_mod():
    return pytest.importorskip("torch")


def test_torch_asl_matches_spec_oracle(torch_mod):
    from common import AsymmetricLoss
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=2.0, size=(8, 12))
    targets = rng.choice([POS, UNK, NEG], size=(8, 12), p=[0.2, 0.6, 0.2])
    for gp in (0.0, 1.0):
        loss = AsymmetricLoss(gamma_neg=4.0, gamma_pos=gp, margin=0.05)(
            torch_mod.tensor(logits, dtype=torch_mod.float32), torch_mod.tensor(targets, dtype=torch_mod.float32))
        assert float(loss) == pytest.approx(asl_spec(logits, targets, gamma_pos=gp), rel=1e-4, abs=1e-6)


def test_torch_asl_legacy_multihot_targets_mean_unknown(torch_mod):
    from common import AsymmetricLoss
    logits = torch_mod.tensor([[2.0, -1.0]])
    multi_hot = torch_mod.tensor([[1.0, 0.0]])
    three = torch_mod.tensor([[1.0, 0.0]])
    f = AsymmetricLoss()
    assert float(f(logits, multi_hot)) == float(f(logits, three))


def test_torch_asl_gradients_finite_and_reject_pushes_down(torch_mod):
    from common import AsymmetricLoss
    x = torch_mod.zeros(1, 3, requires_grad=True)
    AsymmetricLoss()(x, torch_mod.tensor([[1.0, -1.0, 0.0]])).backward()
    assert torch_mod.isfinite(x.grad).all()
    assert x.grad[0, 0] < 0 and x.grad[0, 1] > 0           # positive pulled up, reject pushed down
