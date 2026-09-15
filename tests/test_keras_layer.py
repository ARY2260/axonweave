"""Unit, functional and integration tests for the Keras/TensorFlow ConnectomeLayer."""
import pytest

from tests.conftest import make_graph

tf = pytest.importorskip("tensorflow")
from axonweave.keras import ConnectomeLayer  # noqa: E402


@pytest.fixture
def small_graph():
    return make_graph(n=8, seed=13)


def test_output_shape(small_graph):
    layer = ConnectomeLayer(small_graph)
    layer.build((None, small_graph.n_neurons))
    y = layer(tf.ones((2, small_graph.n_neurons)))
    assert tuple(y.shape) == (2, small_graph.n_neurons)


def test_forward_matches_dense_reference(small_graph):
    """Functional: sparse propagation equals dense reference matmul (y = x @ W)."""
    import numpy as np

    layer = ConnectomeLayer(small_graph)
    layer.build((None, small_graph.n_neurons))
    x = tf.constant(np.random.default_rng(4).random((4, small_graph.n_neurons)).astype("float32"))
    y = layer(x).numpy()
    expected = x.numpy() @ small_graph.weights.toarray()
    assert abs(y - expected).max() < 1e-4


def test_trainable_edges_receive_gradients(small_graph):
    """Functional: trainable edge weights participate in the Keras training loop."""
    layer = ConnectomeLayer(small_graph, trainable_edges=True, learnable_gain=True)
    layer.build((None, small_graph.n_neurons))
    x = tf.ones((2, small_graph.n_neurons))
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(x, training=True))
    grads = tape.gradient(loss, layer.trainable_variables)
    assert grads and all(g is not None for g in grads)


def test_frozen_edges_not_trainable(small_graph):
    layer = ConnectomeLayer(small_graph, trainable_edges=False)
    layer.build((None, small_graph.n_neurons))
    names = [v.name for v in layer.trainable_variables]
    assert not any("edge_weight" in n for n in names)


def test_bias_added(small_graph):
    import numpy as np

    layer = ConnectomeLayer(small_graph, use_bias=True)
    layer.build((None, small_graph.n_neurons))
    y = layer(tf.zeros((2, small_graph.n_neurons))).numpy()
    assert np.count_nonzero(y) == small_graph.n_neurons


def test_wrong_feature_dimension_raises(small_graph):
    layer = ConnectomeLayer(small_graph)
    with pytest.raises(ValueError, match="AXW010"):
        layer.build((None, small_graph.n_neurons + 1))


def test_keras_model_integration(small_graph):
    """Integration: layer composes inside a Keras Sequential model."""
    layer = ConnectomeLayer(small_graph)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(small_graph.n_neurons,)),
        layer,
    ])
    y = model.predict(tf.ones((2, small_graph.n_neurons)), verbose=0)
    assert y.shape == (2, small_graph.n_neurons)


# ---------------------------------------------------------------------------
# API improvements: get_config, AXW010 consistency, signal_policy guard
# ---------------------------------------------------------------------------

def test_get_config_roundtrip(small_graph):
    """Keras serialization: get_config carries the structural options."""
    layer = ConnectomeLayer(
        small_graph, trainable_edges=True, learnable_gain=True, use_bias=True)
    cfg = layer.get_config()
    assert cfg["trainable_edges"] is True
    assert cfg["learnable_gain"] is True
    assert cfg["use_bias"] is True
    assert cfg["n_neurons"] == small_graph.n_neurons


def test_get_config_from_sequential_model(small_graph):
    """Integration: model.get_config() works with the layer embedded."""
    layer = ConnectomeLayer(small_graph, use_bias=True)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(small_graph.n_neurons,)),
        layer,
    ])
    # Must not raise; structural options survive the round trip.
    cfg = model.get_config()
    layer_cfg = cfg["layers"][1]["config"]
    assert layer_cfg["use_bias"] is True


def test_bad_selection_raises_api_usage_error(small_graph):
    from axonweave.errors import ApiUsageError

    with pytest.raises(ApiUsageError, match="AXW010"):
        ConnectomeLayer(small_graph, selection=[1, 2, 3])


def test_signal_policy_warns_axw007(small_graph):
    with pytest.warns(UserWarning, match="AXW007"):
        layer = ConnectomeLayer(small_graph, signal_policy=object())
    assert layer.signal_policy is not None


def test_graph_weights_exposed(small_graph):
    """The backing CSR is reachable (used by KerasConnectomeBlock dynamics)."""
    layer = ConnectomeLayer(small_graph)
    assert layer.graph_weights is small_graph.weights
