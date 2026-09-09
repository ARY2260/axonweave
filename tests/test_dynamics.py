"""Tests for neuron dynamics models (Phase 3)."""
import numpy as np
import pytest
from scipy import sparse

from axonweave.dynamics import AdaptiveLIF, DynamicsPolicy, LIF, Rate
from axonweave.errors import BiologicalAssumptionError


@pytest.fixture
def W():
    return sparse.eye(6, dtype=np.float32, format="csr")


def test_rate_is_identity_like(W):
    d = Rate()
    s = d.initial_state(6)
    x = np.ones(6, dtype=np.float32)
    y, s2 = d.step(s, x, W)
    np.testing.assert_allclose(y, x, rtol=1e-6)


def test_rate_gain_and_baseline():
    d = Rate(gain=2.0, baseline=0.5)
    s = d.initial_state(4)
    y, _ = d.step(s, np.zeros(4, dtype=np.float32), None)
    np.testing.assert_allclose(y, np.full(4, 0.5, dtype=np.float32))


def test_lif_deterministic():
    """Same inputs -> identical spike trains (spec: deterministic simulation)."""
    d = LIF()
    W = sparse.eye(4, dtype=np.float32, format="csr")
    x = np.full(4, 5.0, dtype=np.float32)
    s1 = d.initial_state(4)
    y1, _ = d.step(s1, x, W)
    s2 = d.initial_state(4)
    y2, _ = d.step(s2, x, W)
    np.testing.assert_array_equal(y1, y2)


def test_lif_resting_state_no_spikes(W):
    d = LIF()
    s = d.initial_state(6)
    y, s2 = d.step(s, np.zeros(6, dtype=np.float32), W)
    assert y.sum() == 0
    # v stays at rest
    np.testing.assert_allclose(s2["v"], np.full(6, d.v_rest, dtype=np.float32), atol=1e-6)


def test_lif_strong_input_spikes(W):
    d = LIF(v_rest=-60.0, v_threshold=-55.0)
    s = d.initial_state(6)
    x = np.full(6, 100.0, dtype=np.float32)
    y, _ = d.step(s, x, W)
    assert y.sum() == 6  # every neuron spikes


def test_lif_refractory_blocks_second_spike():
    d = LIF(v_rest=-60.0, v_threshold=-59.0, refractory=5.0, dt=1.0)
    W2 = sparse.eye(2, dtype=np.float32, format="csr")
    s = d.initial_state(2)
    x = np.full(2, 50.0, dtype=np.float32)
    y1, s = d.step(s, x, W2)
    y2, _ = d.step(s, x, W2)  # inside refractory window
    assert y1.sum() == 2
    assert y2.sum() == 0


def test_lif_reset_after_spike():
    d = LIF(v_rest=-60.0, v_threshold=-59.0)
    W1 = sparse.eye(1, dtype=np.float32, format="csr")
    s = d.initial_state(1)
    y, s = d.step(s, np.full(1, 50.0, dtype=np.float32), W1)
    assert y[0] == 1.0
    assert s["v"][0] == d.v_reset


def test_adaptive_lif_threshold_increases():
    d = AdaptiveLIF(v_rest=-60.0, v_threshold=-59.0, tau_adapt=1000.0, delta_threshold=5.0)
    W1 = sparse.eye(1, dtype=np.float32, format="csr")
    s = d.initial_state(1)
    x = np.full(1, 50.0, dtype=np.float32)
    base = d.v_threshold
    y1, s = d.step(s, x, W1)
    assert s["threshold"][0] > base  # threshold jumped after spike


def test_dynamics_policy_override():
    p = DynamicsPolicy(default=Rate(), overrides={"kenyon_cell": LIF()})
    assert isinstance(p.model_for(None), Rate)
    assert isinstance(p.model_for("kenyon_cell"), LIF)
    with pytest.raises(BiologicalAssumptionError, match="AXW005"):
        p.model_for("unknown_type")


def test_batched_state(W):
    d = LIF()
    s = d.initial_state(6, batch_shape=(3,))
    x = np.zeros((3, 6), dtype=np.float32)
    y, s2 = d.step(s, x, W)
    assert y.shape == (3, 6)
    assert s2["v"].shape == (3, 6)
