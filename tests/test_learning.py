"""Tests for learning rules and plasticity (Phase 4)."""
import numpy as np
import pytest
from scipy import sparse

from axonweave.learning import DopamineSTDP, LEARNING_RULES, STDP


@pytest.fixture
def W():
    return sparse.eye(3, dtype=np.float32, format="csr")


def test_registry_has_documented_rules():
    assert set(LEARNING_RULES) == {"stdp", "dopamine_stdp"}


def test_stdp_trace_decay(W):
    rule = STDP(tau_pre=10.0, tau_post=10.0)
    traces = rule.initial_traces(3)
    pre = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rule.update(W, traces, pre, np.zeros(3, dtype=np.float32), dt=1.0, reward=None)
    first = traces["pre"].copy()
    assert first[0] > 0
    rule.update(W, traces, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), dt=1.0)
    assert 0 < traces["pre"][0] < first[0]  # decayed


def test_stdp_changes_existing_edges_only():
    """Topology preservation: W = W0 + dW on existing edges; new edges stay zero."""
    W0 = sparse.eye(3, dtype=np.float32, format="csr")
    rule = STDP(a_plus=0.1, a_minus=0.0)
    traces = rule.initial_traces(3)
    # Pre fires first, then post fires -> potentiation of active synapses.
    rule.update(W0, traces, np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32), dt=1.0)
    rule.update(W0, traces, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32), dt=1.0)
    dense = W0.toarray()
    assert (dense[0, 1] == 0) and (dense[0, 2] == 0)  # no new edges created
    assert dense[0, 0] > 1.0   # existing edge potentiated (LTP)


def test_stdp_weight_clipping():
    rule = STDP(a_plus=1.0, a_minus=0.0, w_min=0.0, w_max=2.0)
    W0 = sparse.eye(3, dtype=np.float32, format="csr") * 1.5
    traces = rule.initial_traces(3)
    ones = np.ones(3, dtype=np.float32)
    rule.update(W0, traces, ones, ones, dt=1.0)
    assert (W0.data <= 2.0).all()


def test_reward_modulated_scales_update():
    rule_a = DopamineSTDP(base=STDP(a_plus=0.1, a_minus=0.0), dopamine_gain=1.0)
    rule_b = DopamineSTDP(base=STDP(a_plus=0.1, a_minus=0.0), dopamine_gain=0.0)
    Wa = sparse.eye(2, dtype=np.float32, format="csr")
    Wb = sparse.eye(2, dtype=np.float32, format="csr")
    ta, tb = rule_a.initial_traces(2), rule_b.initial_traces(2)
    pre, post = np.ones(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
    rule_a.update(Wa, ta, pre, post, dt=1.0, reward=1.0)
    rule_b.update(Wb, tb, pre, post, dt=1.0, reward=1.0)
    rule_a.update(Wa, ta, np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32), dt=1.0, reward=1.0)
    rule_b.update(Wb, tb, np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32), dt=1.0, reward=1.0)
    assert not np.allclose(Wa.data, np.ones(2))     # reward applied
    assert np.allclose(Wb.data, np.ones(2))         # zero gain = no change


def test_plasticity_does_not_touch_original_graph():
    from axonweave.core.brain import BiologicalBrain
    from tests.conftest import make_graph

    brain = BiologicalBrain(make_graph(6, 5))
    before = brain.graph.weights.data.copy()
    rule = STDP(a_plus=0.5)
    W = brain.graph.weights.copy()
    traces = rule.initial_traces(6)
    rule.update(W, traces, np.ones(6, dtype=np.float32), np.ones(6, dtype=np.float32))
    np.testing.assert_array_equal(before, brain.graph.weights.data)
