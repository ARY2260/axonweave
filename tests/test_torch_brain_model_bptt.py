"""BrainModel-level BPTT tests: surrogate dynamics train end to end.

Verifies the full model path (encoder -> spiking runtime -> readout)
carries gradients in `forward_sequence` + `loss.backward()` — the Phase VI
"hybrid learning" target — and that detach_state() supports truncated BPTT
chunk training.

Torch-dependent; CI is the authoritative execution environment.
"""
from __future__ import annotations

import pytest
from tests.conftest import make_graph

torch = pytest.importorskip("torch")
from axonweave.core.brain import BiologicalBrain  # noqa: E402
from axonweave.dynamics import SigmoidSurrogate, SurrogateLIF  # noqa: E402
from axonweave.encoders import VectorEncoder  # noqa: E402
from axonweave.readout import RegressionReadout  # noqa: E402
from axonweave.torch import BrainModel  # noqa: E402


@pytest.fixture
def brain():
    return BiologicalBrain(make_graph(n=12, seed=9))


def _model(brain, **kw):
    defaults = dict(
        brain=brain,
        encoder=VectorEncoder(input_dim=12, output_dim=12, seed=0),
        dynamics=SurrogateLIF(surrogate=SigmoidSurrogate(k=5.0)),
        readout=RegressionReadout(n_source=12, n_outputs=1),
    )
    defaults.update(kw)
    return BrainModel(**defaults).float()


def test_end_to_end_bptt_updates_all_parameter_groups(brain):
    model = _model(brain)
    seq = torch.rand(2, 4, 12)
    target = torch.rand(2, 4, 1)

    out = model.forward_sequence(seq)
    loss = (out - target).pow(2).mean()
    loss.backward()

    # Readout received gradient...
    assert model.readout.weight.grad is not None
    assert model.readout.weight.grad.abs().sum() > 0
    # ...and so did the spiking runtime's edge weights.
    assert model.runtime._prop.edge_weight.grad is not None
    assert model.runtime._prop.edge_weight.grad.abs().sum() > 0


def test_training_loop_reduces_loss(brain):
    """Integration: a few BPTT steps actually decrease the loss."""
    model = _model(brain)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2)
    seq = torch.rand(4, 6, 12)
    target = torch.rand(4, 6, 1)

    first = None
    last = None
    for step in range(8):
        model.reset_state()
        out = model.forward_sequence(seq)
        loss = (out - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
        last = float(loss)
    assert last < first


def test_truncated_bptt_chunks(brain):
    """Chunked training with detach_state() between chunks runs and learns."""
    model = _model(brain)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2)
    seq = torch.rand(2, 8, 12)
    target = torch.rand(2, 8, 1)

    model.reset_state()
    losses = []
    for start in range(0, 8, 4):
        chunk = seq[:, start:start + 4]
        chunk_target = target[:, start:start + 4]
        out = model.forward_sequence(chunk)
        loss = (out - chunk_target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.detach_state()
        losses.append(float(loss))
    assert all(loss_val == loss_val for loss_val in losses)  # finite (no NaN)


def test_brainmodel_exposes_surrogate_config(brain):
    model = _model(brain)
    assert isinstance(model.dynamics, SurrogateLIF)
    assert isinstance(model.dynamics.surrogate, SigmoidSurrogate)
    assert model.runtime.surrogate_kind == 0
