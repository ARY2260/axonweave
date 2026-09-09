"""End-to-end selection tests for BrainModel / ConnectomeBlock / brain.task.

These require torch and run in the CI backend-smoke job. Locally the module
imports are skipped (per spec section 28: write tests, CI executes them).
"""
import numpy as np
import pytest
from tests.conftest import make_graph

torch = pytest.importorskip("torch")
from axonweave.core.brain import BiologicalBrain  # noqa: E402
from axonweave.frameworks.torch import BrainModel, ConnectomeBlock, Input, Readout  # noqa: E402
from axonweave.torch import ConnectomeLayer  # noqa: E402


@pytest.fixture
def brain():
    return BiologicalBrain(make_graph(8, 3))


@pytest.fixture
def sel(brain):
    return brain.graph.neurons.ids([0, 10, 20, 30])


def test_brain_model_selection_sizes(brain, sel):
    """Input/readout projections target the sub-network, not the full brain."""
    model = BrainModel(brain, dynamics="rate", selection=sel)
    assert model.n_active == 4
    assert model.block.n_active == 4
    model.connect(Input(3)).connect(Readout(2))
    assert model.input_proj.out_features == 4
    assert model.readout.in_features == 4


def test_brain_model_forward_on_selection(brain, sel):
    model = BrainModel(brain, dynamics="rate", selection=sel)
    model.connect(Input(3)).connect(Readout(2))
    y = model(torch.randn(5, 3))
    assert y.shape == (5, 2)


def test_selection_block_matches_subgraph_layer(brain, sel):
    """Block output on the selection equals propagation on the sub-graph."""
    block = ConnectomeBlock(brain, dynamics="rate", selection=sel)
    x = torch.randn(2, 4)
    y_block = block(x).numpy()

    sub = type("G", (), {"weights": sel.weights()})
    ref = ConnectomeLayer(sub)
    y_ref = ref(x.numpy())
    np.testing.assert_allclose(y_block, y_ref, rtol=1e-5)


def test_selection_trainable_edges_subset(brain, sel):
    """Trainable edge parameters cover only synapses inside the selection."""
    model = BrainModel(brain, dynamics="rate", trainable_edges=True, selection=sel)
    edge_params = model.block.layer.edge_weight
    assert len(edge_params) == sel.weights().nnz
    # Gradients reach only sub-network edges.
    model.connect(Input(3)).connect(Readout(2))
    model(torch.randn(2, 3)).sum().backward()
    assert edge_params.grad is not None
    assert model.block.layer.selection_body_ids.tolist() == [0, 10, 20, 30]


def test_selection_dimension_error(brain, sel):
    """Wrong input size reports the sub-network dimension, not brain size."""
    block = ConnectomeBlock(brain, dynamics="rate", selection=sel)
    with pytest.raises(ValueError, match=r"AXW010.*\b4\b"):
        block(torch.randn(2, brain.n_neurons))  # full-brain size is wrong here


def test_brain_model_no_selection_unchanged(brain):
    """Backward compatibility: no selection keeps full-brain behavior."""
    model = BrainModel(brain, dynamics="rate")
    model.connect(Input(3)).connect(Readout(2))
    assert model.n_active == brain.n_neurons
    assert model.input_proj.out_features == brain.n_neurons
    y = model(torch.randn(2, 3))
    assert y.shape == (2, 2)


def test_brain_task_selection_passthrough(brain, sel):
    """brain.task(selection=...) flows into BrainModel and its interfaces."""
    from axonweave.encoders import SensorEncoder
    from axonweave.decoders import ActionDecoder

    enc = SensorEncoder(3, 4, seed=1)
    dec = ActionDecoder(actions=2, seed=2)
    model = brain.task(input=enc, output=dec, dynamics="rate", selection=sel,
                       _torch=torch)
    assert model.n_active == 4
    assert model.input_proj.out_features == 4
    assert model.readout.in_features == 4


def test_selection_fit_end_to_end(brain, sel):
    """fit() trains interface params on sub-network data without error."""
    model = BrainModel(brain, dynamics="rate", selection=sel)
    model.connect(Input(3)).connect(Readout(2))
    xs = torch.randn(16, 3)
    ys = torch.randint(0, 2, (16,))
    history = model.fit(list(zip(xs, ys)), epochs=2)
    assert len(history) == 2
    assert all(np.isfinite(h) for h in history)
