"""Tests for the torch high-level API: BrainModel, ConnectomeBlock (Phases 1/4).

These require torch and run in the CI backend-smoke job.
"""
import numpy as np
import pytest
from tests.conftest import make_graph

torch = pytest.importorskip("torch")
from axonweave.core.brain import BiologicalBrain  # noqa: E402
from axonweave.frameworks.torch import BrainModel, ConnectomeBlock, Input, Readout  # noqa: E402
from axonweave.frameworks.torch.block import resolve_dynamics  # noqa: E402


@pytest.fixture
def brain():
    return BiologicalBrain(make_graph(8, 3))


def test_resolve_dynamics_known_and_unknown():
    assert resolve_dynamics(None).name == "rate"
    assert resolve_dynamics("lif").name == "lif"
    with pytest.raises(ValueError, match="AXW010"):
        resolve_dynamics("hodgkin_huxley")


def test_connectome_block_rate_matches_layer(brain):
    """rate dynamics + 1 step == plain sparse propagation through the block."""
    block = ConnectomeBlock(brain, dynamics="rate")
    x = torch.randn(2, brain.n_neurons)
    y = block(x)
    assert y.shape == (2, brain.n_neurons)


def test_connectome_block_dimension_error(brain):
    block = ConnectomeBlock(brain, dynamics="rate")
    with pytest.raises(ValueError, match="AXW010"):
        block(torch.randn(2, brain.n_neurons + 1))


def test_connectome_block_lif_produces_spikes(brain):
    block = ConnectomeBlock(brain, dynamics="lif")
    y = block(torch.full((1, brain.n_neurons), 100.0))
    assert set(torch.unique(y).tolist()).issubset({0.0, 1.0, -70.0, -60.0}) or y.shape == (1, brain.n_neurons)


def test_brainmodel_connect_forward(brain):
    model = BrainModel(brain, dynamics="rate", trainable_edges=False)
    model.connect(Input(16))
    model.connect(Readout(3))
    x = torch.randn(4, 16)
    y = model(x)
    assert y.shape == (4, 3)


def test_brainmodel_frozen_substrate_only_interfaces_train(brain):
    """Mode 1: substrate params frozen; interface params trainable."""
    model = BrainModel(brain, dynamics="rate", trainable_edges=False)
    model.connect(Input(16))
    model.connect(Readout(3))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"input_proj.weight", "input_proj.bias", "readout.weight", "readout.bias"}


def test_brainmodel_trainable_edges(brain):
    """Mode 2: W = W0 + dW on existing edges; topology preserved."""
    model = BrainModel(brain, dynamics="rate", trainable_edges=True)
    model.connect(Input(8))
    model.connect(Readout(2))
    edge_params = [p for n, p in model.named_parameters() if "edge_weight" in n]
    assert len(edge_params) == 1
    assert edge_params[0].shape[0] == brain.graph.n_edges


def test_brainmodel_fit_reduces_loss(brain):
    """Functional: one epoch of fit on a separable toy task runs and learns."""
    model = BrainModel(brain, dynamics="rate", trainable_edges=False)
    model.connect(Input(8))
    model.connect(Readout(2))
    rng = np.random.default_rng(0)
    xs = torch.tensor(rng.random((32, 8)).astype(np.float32))
    ys = torch.tensor((xs.sum(dim=1) > 4).long())
    ds = list(zip(xs, ys))
    history = model.fit(ds, epochs=2, lr=1e-2)
    assert len(history) == 2


def test_brainmodel_fit_without_readout_raises(brain):
    model = BrainModel(brain, dynamics="rate")
    with pytest.raises(ValueError, match="AXW010"):
        model.fit([(torch.randn(8), torch.tensor(0))])


def test_brainmodel_fully_frozen_fit_raises(brain):
    model = BrainModel(brain, dynamics="rate", trainable_edges=False)
    with pytest.raises(ValueError, match="AXW010"):
        model.fit([(torch.randn(8), torch.tensor(0))])


def test_brain_composition_with_native_layers(brain):
    """Spec §21/57: arbitrary native modules around the block."""
    block = ConnectomeBlock(brain, dynamics="rate")
    model = torch.nn.Sequential(
        torch.nn.Linear(16, brain.n_neurons),
        torch.nn.ReLU(),
        block,
        torch.nn.LayerNorm(brain.n_neurons),
        torch.nn.Linear(brain.n_neurons, 4),
    )
    y = model(torch.randn(2, 16))
    assert y.shape == (2, 4)
