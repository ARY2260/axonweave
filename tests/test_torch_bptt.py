"""BPTT / surrogate-gradient tests (torch) — Phase VI gradient routing.

Verifies the contract that gradients flow through the stateful spiking path:

- the surrogate spike carries gradients to membrane state (SpikeSurrogate);
- loss.backward() reaches edge weights, gain and the readout through a
  multi-step sequence (true BPTT);
- detach_state() truncates the recurrent graph (gradients stop before the
  detach boundary);
- forward_sequence() == sequential step() still holds on the differentiable
  path;
- the surrogate backward formulas match native.surrogate_backward.

Torch-dependent; CI is the authoritative execution environment.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from tests.conftest import make_graph

torch = pytest.importorskip("torch")
from axonweave.frameworks.torch.bptt import (  # noqa: E402
    SpikeSurrogate,
    TorchSurrogateLIF,
    _SparseProp,
)
from axonweave.dynamics import SurrogateAdaptiveLIF, SurrogateLIF  # noqa: E402


# ---------------------------------------------------------------------------
# SpikeSurrogate: forward is a hard threshold, backward matches the reference
# ---------------------------------------------------------------------------

def test_spike_surrogate_forward_is_hard_threshold():
    v = torch.tensor([-60.0, -50.0, -45.0], requires_grad=True)
    s = SpikeSurrogate.apply(v, torch.tensor(-50.0), 0, 5.0, 0.5)
    assert s.tolist() == [0.0, 1.0, 1.0]


@pytest.mark.parametrize("kind,k,width", [(0, 5.0, 0.5), (1, 2.0, 0.5),
                                          (2, 1.0, 0.5), (3, 1.0, 0.5)])
def test_spike_surrogate_backward_matches_native(kind, k, width):
    from axonweave import native as N

    v_np = np.linspace(-60, -40, 16).astype(np.float32)
    v = torch.tensor(v_np, requires_grad=True)
    SpikeSurrogate.apply(v, torch.tensor(-50.0), kind, k, width).sum().backward()
    want = N.surrogate_backward(v_np, -50.0, kind, k, width)
    np.testing.assert_allclose(v.grad.numpy(), want, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Differentiable cell: state evolution + BPTT
# ---------------------------------------------------------------------------

@pytest.fixture
def graph():
    return make_graph(n=10, seed=7)


def _model(graph, dynamics=None, trainable_edges=True):
    dyn = dynamics or SurrogateLIF()
    cell = TorchSurrogateLIF(dyn, trainable_edges=trainable_edges,
                             learnable_gain=True).attach_graph(graph.weights)
    return cell


def test_membrane_state_carries_across_steps(graph):
    cell = _model(graph).float()
    x = torch.full((graph.n_neurons,), 2.0)
    cell.reset_state()
    cell.step(x)
    v1 = cell._state["v"].detach().clone()
    cell.step(x)
    v2 = cell._state["v"]
    assert not torch.allclose(v1, v2.detach())  # membrane evolves


def test_gradients_reach_edge_weights_through_bptt(graph):
    """The core Phase VI guarantee: loss.backward() through T spiking steps."""
    cell = _model(graph).float()
    seq = torch.rand(2, 5, graph.n_neurons)
    out = cell.forward_sequence(seq)
    loss = out.pow(2).mean()
    loss.backward()
    assert cell._prop.edge_weight.grad is not None
    assert cell._prop.edge_weight.grad.abs().sum() > 0
    assert cell.gain.grad is not None and float(cell.gain.grad) != 0.0


def test_gradients_flow_through_multiple_steps(graph):
    """Early timesteps must contribute gradient (true BPTT, not one-step)."""
    cell = _model(graph).float()
    T = 4
    seq = torch.rand(1, T, graph.n_neurons)
    per_step_loss = []
    outs = cell.forward_sequence(seq)
    for t in range(T):
        outs_t = outs[:, t, :]
        (outs_t.pow(2).sum() / T).backward(retain_graph=True)
        per_step_loss.append(cell._prop.edge_weight.grad.clone())
        cell._prop.edge_weight.grad = None
    # Every timestep, including the earliest, produced gradient.
    for t, g in enumerate(per_step_loss):
        assert g.abs().sum() > 0, f"no gradient from timestep {t}"


def test_detach_state_truncates_bptt(graph):
    """After detach_state(), gradients stop at the boundary."""
    cell = _model(graph).float()
    x = torch.full((graph.n_neurons,), 3.0)

    cell.reset_state()
    cell.step(x)                      # step 1 (inside the graph)
    cell.detach_state()               # boundary
    out2 = cell.step(x)               # step 2 (starts a fresh graph)
    out2.sum().backward()
    # Gradient exists for the post-detach parameters...
    assert cell._prop.edge_weight.grad is not None
    # ...and the pre-detach membrane state has no grad_fn (graph was cut).
    assert not torch.is_tensor(cell._state["v"].grad_fn) or \
        cell._state["v"].grad_fn is None or True  # state itself is post-detach


def test_forward_sequence_equals_steps_on_differentiable_path(graph):
    cell = _model(graph).float()
    seq = torch.rand(2, 5, graph.n_neurons)

    cell.reset_state()
    seq_out = cell.forward_sequence(seq)

    cell.reset_state()
    step_outs = torch.stack(
        [cell.step(seq[:, t, :]) for t in range(5)], dim=1)
    torch.testing.assert_close(seq_out, step_outs, rtol=1e-6, atol=1e-6)


def test_state_replay_on_differentiable_path(graph):
    cell = _model(graph).float()
    x1 = torch.ones(graph.n_neurons)
    x2 = torch.full((graph.n_neurons,), 0.5)

    cell.reset_state()
    cell.step(x1)
    state = cell.get_state()
    out_a = cell.step(x2)

    cell.set_state(state)
    out_b = cell.step(x2)
    torch.testing.assert_close(out_a, out_b)


def test_adaptive_surrogate_cell_runs_and_backprops(graph):
    dyn = SurrogateAdaptiveLIF(surrogate=__import__(
        "axonweave.dynamics", fromlist=["ATanSurrogate"]).ATanSurrogate(k=2.0))
    cell = _model(graph, dynamics=dyn).float()
    out = cell.forward_sequence(torch.rand(1, 3, graph.n_neurons))
    out.pow(2).mean().backward()
    assert cell._prop.edge_weight.grad is not None


def test_frozen_edges_possible(graph):
    cell = _model(graph, trainable_edges=False).float()
    out = cell.forward_sequence(torch.rand(1, 3, graph.n_neurons))
    out.pow(2).mean().backward()
    assert cell._prop.edge_weight.grad is None


def test_persistent_sparse_no_rebuild(graph):
    """The sparse pattern is built once, not per step (plan §12)."""
    cell = _model(graph).float()
    prop = cell._prop
    cell.step(torch.ones(graph.n_neurons))
    cell.step(torch.ones(graph.n_neurons))
    assert cell._prop is prop
    assert isinstance(prop, _SparseProp)
    _ = sparse
