"""Tests for the temporal connectome runtime (Temporal Runtime Alpha).

Covers the backend-neutral semantics contract:
- step() persists state and never auto-resets;
- forward_sequence() == sequential step() from the same initial state;
- get_state()/set_state() replay is exact;
- RuntimeState snapshot copies are deep;
- streaming use and shape handling for [B, F] and [B, T, F].
"""
from __future__ import annotations

import numpy as np
import pytest

from axonweave.dynamics import AdaptiveLIF, LIF, Rate
from axonweave.errors import ApiUsageError
from axonweave.runtime import (
    ConnectomeRuntime,
    NeuronState,
    PlasticityState,
    RuntimeState,
    SynapticState,
)

from tests.conftest import make_graph


@pytest.fixture
def graph():
    return make_graph(n=12, seed=3)


def _stream(n_steps=5, n=12, batch=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((batch, n_steps, n)).astype(np.float32)


# ---------------------------------------------------------------------------
# step semantics
# ---------------------------------------------------------------------------

def test_step_persists_state_never_auto_resets(graph):
    rt = ConnectomeRuntime(graph, dynamics=Rate())
    x = np.ones(12, dtype=np.float32)
    y1 = rt.step(x)
    y2 = rt.step(x)
    assert rt.timestep == 2
    # Rate dynamics are stateless per step, but the clock must advance.
    assert rt.get_state().timestep == 2
    np.testing.assert_array_equal(y1, y2)


def test_step_accepts_1d_and_batched(graph):
    rt = ConnectomeRuntime(graph, dynamics=Rate())
    y1 = rt.step(np.ones(12, dtype=np.float32))
    assert y1.shape == (12,)
    y2 = rt.step(np.ones((3, 12), dtype=np.float32))
    assert y2.shape == (3, 12)


def test_step_wrong_dimension_raises(graph):
    rt = ConnectomeRuntime(graph)
    with pytest.raises(ApiUsageError, match="AXW010"):
        rt.step(np.ones(13, dtype=np.float32))


def test_lif_state_carries_between_steps(graph):
    """Functional: a constant input drives LIF membrane toward threshold."""
    rt = ConnectomeRuntime(graph, dynamics=LIF(dt=1.0))
    x = np.full(graph.n_neurons, 2.0, dtype=np.float32)
    rt.step(x)
    v_after_1 = rt.get_state().neuron.variables["v"].copy()
    rt.step(x)
    v_after_2 = rt.get_state().neuron.variables["v"].copy()
    # Membrane must evolve, not restart from rest each call.
    assert not np.allclose(v_after_1, v_after_2)


# ---------------------------------------------------------------------------
# sequence equivalence (the central contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dynamics", [Rate(), LIF(), AdaptiveLIF()],
                         ids=["rate", "lif", "adaptive_lif"])
def test_forward_sequence_equals_sequential_steps(graph, dynamics):
    """The core guarantee: one call == T step() calls from the same state."""
    seq = _stream(n_steps=6, n=graph.n_neurons)

    rt_seq = ConnectomeRuntime(graph, dynamics=dynamics)
    out_seq = rt_seq.forward_sequence(seq.copy())

    rt_step = ConnectomeRuntime(graph, dynamics=dynamics)
    outs = [rt_step.step(seq[:, t, :]) for t in range(seq.shape[1])]
    out_steps = np.stack(outs, axis=1)

    np.testing.assert_allclose(out_seq, out_steps, rtol=1e-6, atol=1e-6)
    assert rt_seq.timestep == rt_step.timestep == seq.shape[1]


def test_forward_sequence_single_sample_no_batch(graph):
    seq = _stream(batch=1, n_steps=4)
    rt = ConnectomeRuntime(graph, dynamics=Rate())
    out = rt.forward_sequence(seq[0])  # [T, F]
    assert out.shape == (4, graph.n_neurons)


def test_forward_sequence_shape_validation(graph):
    rt = ConnectomeRuntime(graph)
    with pytest.raises(ApiUsageError, match="AXW010"):
        rt.forward_sequence(np.ones((12,), dtype=np.float32))
    with pytest.raises(ApiUsageError, match="AXW010"):
        rt.forward_sequence(np.ones((2, 4, 13), dtype=np.float32))


# ---------------------------------------------------------------------------
# state capture / replay / branching
# ---------------------------------------------------------------------------

def test_get_set_state_replays_exactly(graph):
    """The milestone replay contract: same input after state restore -> same output."""
    rt = ConnectomeRuntime(graph, dynamics=LIF())
    x1 = np.ones(12, dtype=np.float32)
    x2 = np.full(12, 0.5, dtype=np.float32)

    rt.reset_state()
    y_a1 = rt.step(x1)
    state = rt.get_state()
    y_a2 = rt.step(x2)

    # Replay the branch from the captured state.
    rt.set_state(state)
    y_b2 = rt.step(x2)

    np.testing.assert_array_equal(y_a2, y_b2)
    # And the first-step output is unchanged by the branching.
    rt.set_state(state)
    np.testing.assert_array_equal(rt.step(x1), y_a1)
    _ = y_a2


def test_state_snapshot_is_deep(graph):
    rt = ConnectomeRuntime(graph, dynamics=LIF())
    rt.step(np.ones(12, dtype=np.float32))
    state = rt.get_state()
    v_before = state.neuron.variables["v"].copy()
    rt.step(np.full(12, 5.0, dtype=np.float32))
    # The captured snapshot must not have been mutated by later steps.
    np.testing.assert_array_equal(state.neuron.variables["v"], v_before)


def test_set_state_rejects_foreign_objects(graph):
    rt = ConnectomeRuntime(graph)
    with pytest.raises(ApiUsageError, match="AXW010"):
        rt.set_state({"v": np.zeros(3)})


def test_state_roundtrip_serializable(graph):
    """RuntimeState.to_dict() round-trips through plain structures."""
    rt = ConnectomeRuntime(graph, dynamics=LIF())
    rt.step(np.ones(12, dtype=np.float32))
    d = rt.get_state().to_dict()
    assert d["timestep"] == 1
    assert "v" in d["neuron"] and "refrac_until" in d["neuron"]
    # A fresh runtime restored from the dict-shaped state behaves the same.
    state = RuntimeState(
        timestep=d["timestep"],
        neuron=NeuronState(variables=d["neuron"]),
        synaptic=SynapticState(variables := d["synaptic"]),
        plasticity=PlasticityState(d["plasticity"]),
    )
    rt2 = ConnectomeRuntime(graph, dynamics=LIF())
    rt2.set_state(state)
    np.testing.assert_array_equal(
        rt2.step(np.ones(12, dtype=np.float32)),
        rt.step(np.ones(12, dtype=np.float32)),
    )
    _ = variables


def test_detach_state_returns_self(graph):
    rt = ConnectomeRuntime(graph)
    assert rt.detach_state() is rt


def test_reset_state_clears_everything(graph):
    rt = ConnectomeRuntime(graph, dynamics=LIF())
    rt.step(np.ones(12, dtype=np.float32))
    rt.reset_state()
    assert rt.timestep == 0
    assert rt.get_state().timestep == 0


# ---------------------------------------------------------------------------
# construction / memory
# ---------------------------------------------------------------------------

def test_runtime_requires_graph_or_n_neurons():
    with pytest.raises(ApiUsageError, match="AXW010"):
        ConnectomeRuntime()


def test_runtime_graphfree_mode():
    rt = ConnectomeRuntime(n_neurons=6, dynamics=Rate())
    y = rt.step(np.ones(6, dtype=np.float32))
    assert y.shape == (6,)


def test_runtime_rejects_non_dynamics(graph):
    with pytest.raises(ApiUsageError, match="AXW010"):
        ConnectomeRuntime(graph, dynamics="lif")  # str not allowed here


def test_memory_estimate_components(graph):
    rt = ConnectomeRuntime(graph, dynamics=LIF())
    rt.step(np.ones(12, dtype=np.float32))
    est = rt.memory_estimate()
    assert est["neuron_state"] > 0          # v + refrac arrays
    assert est["edge_parameters"] == graph.weights.nnz * 4  # float32
    assert est["delay_buffer"] == 0
    assert est["dtype"] == "float32"


def test_brain_memory_estimate():
    """brain.memory_estimate() reports per-component bytes."""
    from axonweave.core.brain import BiologicalBrain
    from scipy import sparse

    g = make_graph(n=10, seed=1)
    brain = BiologicalBrain(g)
    est = brain.memory_estimate()
    assert est["n_neurons"] == 10
    assert est["edge_parameters"] == g.weights.nnz * 4
    assert est["neuron_state"] > 0
    assert "total" in est
    # float64 doubles the neuron-state estimate.
    est64 = brain.memory_estimate(dtype="float64")
    assert est64["neuron_state"] == 2 * est["neuron_state"]
    _ = sparse
