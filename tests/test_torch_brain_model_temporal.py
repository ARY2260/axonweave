"""Stateful BrainModel tests (torch) — Temporal Runtime Alpha milestone.

Written against the milestone contract from the development plan:

    model.reset_state()
    for timestep in stream:
        prediction = model.step(timestep)

    model.forward_sequence(sequence) == sequential step() calls
    state = model.get_state(); model.step(x); model.set_state(state)

These require torch; CI is the authoritative execution environment.
"""
from __future__ import annotations

import pytest

from tests.conftest import make_graph

torch = pytest.importorskip("torch")
from axonweave.frameworks.torch.runtime_bridge import TorchStatefulRuntime  # noqa: E402
from axonweave.torch import BrainModel  # noqa: E402


@pytest.fixture
def graph():
    return make_graph(n=12, seed=5)


def test_runtime_bridge_step_persists(graph):
    rt = TorchStatefulRuntime(graph, dynamics=__import__(
        "axonweave.dynamics", fromlist=["LIF"]).LIF())
    x = torch.ones(graph.n_neurons)
    rt.step(x)
    assert rt.timestep == 1
    rt.step(x)
    assert rt.timestep == 2


def test_runtime_bridge_sequence_equals_steps(graph):
    from axonweave.dynamics import LIF

    seq = torch.rand(2, 6, graph.n_neurons)
    rt_a = TorchStatefulRuntime(graph, dynamics=LIF())
    out_a = rt_a.forward_sequence(seq)
    rt_b = TorchStatefulRuntime(graph, dynamics=LIF())
    outs = [rt_b.step(seq[:, t, :]) for t in range(6)]
    out_b = torch.stack(outs, dim=1)
    torch.testing.assert_close(out_a, out_b, rtol=1e-5, atol=1e-5)


def test_brain_model_constructor_injection(graph):
    """The milestone wiring: BrainModel(brain=, encoder=, dynamics=, readout=)."""
    from axonweave.dynamics import Rate
    from axonweave.encoders import VectorEncoder
    from axonweave.readout import RegressionReadout

    model = BrainModel(
        brain=graph_brain(graph),
        encoder=VectorEncoder(input_dim=8, output_dim=12, seed=0),
        dynamics=Rate(),
        readout=RegressionReadout(n_source=12, n_outputs=1),
    )
    assert model.encoder is not None
    assert model.dynamics is not None
    assert model.readout is not None
    assert model.runtime is not None
    y = model(torch.randn(2, 8))
    assert y.shape == (2, 1)


def test_brain_model_streaming_step_semantics(graph):
    from axonweave.dynamics import Rate
    from axonweave.encoders import VectorEncoder
    from axonweave.readout import RegressionReadout

    model = BrainModel(
        brain=graph_brain(graph),
        encoder=VectorEncoder(input_dim=12, output_dim=12, seed=1),
        dynamics=Rate(),
        readout=RegressionReadout(n_source=12, n_outputs=2),
    ).float()
    model.reset_state()
    preds = [model.step(torch.ones(12)) for _ in range(4)]
    assert model.runtime.timestep == 4
    # State persistence: identical inputs after reset produce identical steps.
    model.reset_state()
    preds2 = [model.step(torch.ones(12)) for _ in range(4)]
    for p1, p2 in zip(preds, preds2):
        torch.testing.assert_close(p1, p2)


def test_brain_model_sequence_step_equivalence(graph):
    from axonweave.dynamics import LIF
    from axonweave.encoders import VectorEncoder
    from axonweave.readout import RegressionReadout

    model = BrainModel(
        brain=graph_brain(graph),
        encoder=VectorEncoder(input_dim=12, output_dim=12, seed=2),
        dynamics=LIF(),
        readout=RegressionReadout(n_source=12, n_outputs=1),
    ).float()
    seq = torch.rand(3, 5, 12)

    model.reset_state()
    seq_out = model.forward_sequence(seq)

    model.reset_state()
    step_outs = torch.stack(
        [model.step(seq[:, t, :]) for t in range(5)], dim=1)

    torch.testing.assert_close(seq_out, step_outs, rtol=1e-5, atol=1e-5)


def test_brain_model_state_replay(graph):
    from axonweave.dynamics import LIF
    from axonweave.encoders import VectorEncoder
    from axonweave.readout import RegressionReadout

    model = BrainModel(
        brain=graph_brain(graph),
        encoder=VectorEncoder(input_dim=12, output_dim=12, seed=3),
        dynamics=LIF(),
        readout=RegressionReadout(n_source=12, n_outputs=1),
    ).float()
    x1, x2 = torch.ones(12), torch.full((12,), 0.5)

    model.reset_state()
    model.step(x1)
    state = model.get_state()
    out_a = model.step(x2)

    model.set_state(state)
    out_b = model.step(x2)
    torch.testing.assert_close(out_a, out_b)


def graph_brain(graph):
    from axonweave.core.brain import BiologicalBrain

    return BiologicalBrain(graph)
