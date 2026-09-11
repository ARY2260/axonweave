"""Tests for surrogate-gradient spiking dynamics.

Verifies that SurrogateLIF / SurrogateAdaptiveLIF keep the deterministic
spike decisions of their parent dynamics while exposing a stored
surrogate gradient via ``state["spike_gradient"]`` and the
``spike_gradient`` property.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from axonweave.dynamics import (
    AdaptiveLIF,
    LIF,
    SurrogateAdaptiveLIF,
    SurrogateLIF,
)
from axonweave.dynamics.surrogate import SigmoidSurrogate, StraightThroughEstimator


@pytest.fixture
def W():
    return sparse.eye(4, dtype=np.float32, format="csr")


def _ste_lif(**kwargs):
    defaults = dict(v_rest=0.0, v_threshold=1.0, v_reset=0.0, refractory=0.0, tau=1.0, dt=1.0)
    defaults.update(kwargs)
    return SurrogateLIF(surrogate=StraightThroughEstimator(width=0.5), **defaults)


def test_is_lif_subclass():
    assert issubclass(SurrogateLIF, LIF)
    assert issubclass(SurrogateAdaptiveLIF, AdaptiveLIF)


def test_surrogate_names():
    assert SurrogateLIF().name == "surrogate_lif"
    assert SurrogateAdaptiveLIF().name == "surrogate_adaptive_lif"
    assert isinstance(SurrogateLIF().surrogate, SigmoidSurrogate)


def test_default_spike_gradient_is_none():
    assert SurrogateLIF().spike_gradient is None
    assert SurrogateAdaptiveLIF().spike_gradient is None


def test_spike_decision_matches_threshold_crossing(W):
    d = _ste_lif()
    s = d.initial_state(4)
    x = np.full(4, 1.0, dtype=np.float32)
    y, s2 = d.step(s, x, W)
    np.testing.assert_array_equal(y, np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(s2["v"], np.zeros(4, dtype=np.float32))


def test_stored_gradient_matches_property(W):
    d = _ste_lif()
    s = d.initial_state(4)
    x = np.full(4, 1.0, dtype=np.float32)
    _, s2 = d.step(s, x, W)
    np.testing.assert_array_equal(s2["spike_gradient"], np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(d.spike_gradient, np.ones(4, dtype=np.float32))


def test_gradient_masked_in_refractory():
    d = _ste_lif(refractory=5.0)
    W1 = sparse.eye(2, dtype=np.float32, format="csr")
    s = d.initial_state(2)
    x = np.full(2, 1.0, dtype=np.float32)
    y1, s = d.step(s, x, W1)
    y2, s = d.step(s, x, W1)
    assert y1.sum() == 2
    assert y2.sum() == 0
    np.testing.assert_array_equal(s["spike_gradient"], np.zeros(2, dtype=np.float32))


def test_subthreshold_input_still_differentiable(W):
    d = _ste_lif()
    s = d.initial_state(4)
    x = np.full(4, 0.5, dtype=np.float32)
    y, s2 = d.step(s, x, W)
    assert y.sum() == 0
    # v stays at 0.5, well inside the STE window -> nonzero gradient.
    np.testing.assert_array_equal(s2["spike_gradient"], np.ones(4, dtype=np.float32))


def test_batched_shape(W):
    d = _ste_lif()
    s = d.initial_state(4, batch_shape=(3,))
    x = np.ones((3, 4), dtype=np.float32)
    y, s2 = d.step(s, x, W)
    assert y.shape == (3, 4)
    assert s2["spike_gradient"].shape == (3, 4)
    assert d.spike_gradient.shape == (3, 4)


def test_sigmoid_gradient_value_at_threshold(W):
    d = SurrogateLIF(surrogate=SigmoidSurrogate(k=5.0),
                     v_rest=0.0, v_threshold=1.0, v_reset=0.0, tau=1.0, dt=1.0)
    s = d.initial_state(4)
    x = np.full(4, 1.0, dtype=np.float32)
    _, s2 = d.step(s, x, W)
    np.testing.assert_allclose(s2["spike_gradient"], np.full(4, 1.25, dtype=np.float32))


def test_gradient_zero_when_far_below_threshold(W):
    d = SurrogateLIF(surrogate=SigmoidSurrogate(k=5.0),
                     v_rest=-65.0, v_threshold=-50.0, tau=20.0, dt=1.0)
    s = d.initial_state(4)
    x = np.zeros(4, dtype=np.float32)
    _, s2 = d.step(s, x, W)
    np.testing.assert_allclose(s2["spike_gradient"], np.zeros(4, dtype=np.float32), atol=1e-5)


def test_adaptive_threshold_adaptation_preserved():
    d = SurrogateAdaptiveLIF(surrogate=StraightThroughEstimator(width=0.5),
                             v_rest=0.0, v_threshold=1.0, v_reset=0.0,
                             tau=1.0, dt=1.0, tau_adapt=1.0, delta_threshold=1.0)
    W1 = sparse.eye(1, dtype=np.float32, format="csr")
    s = d.initial_state(1)
    x = np.full(1, 1.0, dtype=np.float32)
    y, s2 = d.step(s, x, W1)
    assert y[0] == 1.0
    assert s2["threshold"][0] > 1.0
    np.testing.assert_allclose(s2["spike_gradient"], np.array([1.0], dtype=np.float32))


def test_adaptive_gradient_zero_inside_refractory():
    d = SurrogateAdaptiveLIF(surrogate=StraightThroughEstimator(width=0.5),
                             v_rest=0.0, v_threshold=1.0, v_reset=0.0,
                             tau=1.0, dt=1.0, tau_adapt=2.0, delta_threshold=0.5,
                             refractory=5.0)
    W1 = sparse.eye(1, dtype=np.float32, format="csr")
    s = d.initial_state(1)
    x = np.full(1, 1.0, dtype=np.float32)
    _, s = d.step(s, x, W1)
    _, s2 = d.step(s, x, W1)
    assert s2["spike_gradient"][0] == 0.0


def test_initial_state_matches_lif():
    parent = LIF()
    child = SurrogateLIF()
    np.testing.assert_allclose(child.initial_state(3)["v"], parent.initial_state(3)["v"])