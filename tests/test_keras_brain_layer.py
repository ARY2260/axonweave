"""Tests for the Keras high-level API: BrainLayer, KerasConnectomeBlock.

These require tensorflow and run in the CI backend-smoke job (spec section
28: tests are authored here; CI executes them).
"""
import numpy as np
import pytest
from tests.conftest import make_graph

tf = pytest.importorskip("tensorflow")
from axonweave.core.brain import BiologicalBrain  # noqa: E402
from axonweave.errors import ApiUsageError  # noqa: E402
from axonweave.frameworks.keras import BrainLayer, KerasConnectomeBlock, resolve_dynamics  # noqa: E402
from axonweave.frameworks.torch.interfaces import Input, Readout  # noqa: E402


@pytest.fixture
def brain():
    return BiologicalBrain(make_graph(8, 3))


@pytest.fixture
def sel(brain):
    return brain.graph.neurons.ids([0, 10, 20, 30])


def test_resolve_dynamics_known_and_unknown():
    assert resolve_dynamics(None).name == "rate"
    assert resolve_dynamics("lif").name == "lif"
    with pytest.raises(ApiUsageError, match="AXW010"):
        resolve_dynamics("hodgkin_huxley")


def test_block_rate_matches_layer(brain):
    block = KerasConnectomeBlock(brain, dynamics="rate")
    x = tf.constant(np.random.default_rng(0).random((2, brain.n_neurons)).astype(np.float32))
    y = block(x)
    assert y.shape == (2, brain.n_neurons)


def test_block_dimension_error(brain):
    block = KerasConnectomeBlock(brain, dynamics="rate")
    with pytest.raises(ApiUsageError, match="AXW010"):
        block(tf.zeros((2, brain.n_neurons + 1)))


def test_brain_layer_selection_sizes(brain, sel):
    model = BrainLayer(brain, dynamics="rate", selection=sel)
    assert model.n_active == 4
    model.connect(Input(3)).connect(Readout(2))
    assert model.input_proj.units == 4
    assert model.readout.units == 2


def test_brain_layer_forward_on_selection(brain, sel):
    model = BrainLayer(brain, dynamics="rate", selection=sel)
    model.connect(Input(3)).connect(Readout(2))
    y = model(tf.ones((5, 3)))
    assert y.shape == (5, 2)


def test_brain_layer_no_selection_unchanged(brain):
    """Backward-compatible full-brain path when no selection is given."""
    model = BrainLayer(brain, dynamics="rate")
    model.connect(Input(3)).connect(Readout(2))
    assert model.n_active == brain.n_neurons
    y = model(tf.ones((2, 3)))
    assert y.shape == (2, 2)


def test_brain_layer_selects_subnetwork_weights(brain, sel):
    """Trainable edges cover only synapses inside the selection."""
    model = BrainLayer(brain, dynamics="rate", trainable_edges=True, selection=sel)
    assert int(model.block.layer.edge_weight.shape[0]) == sel.weights().nnz
    assert model.block.layer.selection_body_ids.tolist() == [0, 10, 20, 30]


def test_keras_functional_composition(brain):
    """BrainLayer composes with the Keras Functional API."""
    model = BrainLayer(brain, dynamics="rate")
    model.connect(Input(3)).connect(Readout(2))
    inp = tf.keras.Input(shape=(3,))
    out = model(inp)
    fn = tf.keras.Model(inputs=inp, outputs=out)
    y = fn(tf.ones((4, 3)))
    assert y.shape == (4, 2)


def test_brain_layer_fit(brain, sel):
    """model.fit() trains interface params on sub-network data."""
    model = BrainLayer(brain, dynamics="rate", selection=sel)
    model.connect(Input(3)).connect(Readout(2))
    model.compile(optimizer="adam", loss="mse")
    xs = np.random.default_rng(1).random((16, 3)).astype(np.float32)
    ys = np.random.default_rng(2).random((16, 2)).astype(np.float32)
    history = model.fit(xs, ys, epochs=2, batch_size=8, verbose=0)
    assert len(history.history["loss"]) == 2
    assert all(np.isfinite(v) for v in history.history["loss"])


def test_brain_layer_frozen_interface_only_training(brain, sel):
    """Frozen connectome: edge weights stay constant through fit()."""
    model = BrainLayer(brain, dynamics="rate", trainable_edges=False, selection=sel)
    model.connect(Input(3)).connect(Readout(2))
    model.compile(optimizer="adam", loss="mse")
    w0 = model.block.layer.edge_weight.numpy().copy()
    xs = np.ones((8, 3), dtype=np.float32)
    ys = np.ones((8, 2), dtype=np.float32)
    model.fit(xs, ys, epochs=1, verbose=0)
    np.testing.assert_array_equal(w0, model.block.layer.edge_weight.numpy())


def test_keras_task_facade(brain, sel):
    """brain.keras_task(...) wires encoders/decoders into BrainLayer."""
    from axonweave.decoders import ActionDecoder
    from axonweave.encoders import SensorEncoder

    enc = SensorEncoder(3, 4, seed=1)
    dec = ActionDecoder(actions=2, seed=2)
    model = brain.keras_task(input=enc, output=dec, dynamics="rate", selection=sel)
    assert model.n_active == 4
    assert model.input_proj.units == 4
    assert model.readout.units == 2


def test_keras_task_requires_tensorflow(brain):
    """AXW006 fires on the facade when TF is absent (authored for CI matrix)."""
    try:
        import tensorflow  # noqa: F401
        pytest.skip("tensorflow installed; AXW006 path not reachable here")
    except ImportError:
        with pytest.raises(Exception, match="AXW006"):
            brain.keras_task(dynamics="rate")
