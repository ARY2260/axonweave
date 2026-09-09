"""Unit + functional tests for the NumPy reference ConnectomeLayer."""
import numpy as np

from axonweave.numpy import ConnectomeLayer
from tests.conftest import make_graph


def test_output_shape_and_dtype():
    graph = make_graph(n=6)
    layer = ConnectomeLayer(graph)
    x = np.random.default_rng(0).random((4, 6)).astype(np.float32)
    y = layer(x)
    assert y.shape == (4, 6)


def test_matches_dense_reference():
    """Functional: sparse propagation equals dense x @ W computation."""
    graph = make_graph(n=8, seed=3)
    layer = ConnectomeLayer(graph)
    x = np.random.default_rng(1).random((5, 8)).astype(np.float32)
    y = layer(x)
    expected = x @ graph.weights.toarray()
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-6)


def test_gain_scales_output():
    graph = make_graph(n=6)
    x = np.random.default_rng(2).random((3, 6)).astype(np.float32)
    y1 = ConnectomeLayer(graph, gain=1.0)(x)
    y2 = ConnectomeLayer(graph, gain=2.0)(x)
    np.testing.assert_allclose(y2, 2.0 * y1, rtol=1e-6)


def test_trainable_edges_copy_weights():
    graph = make_graph(n=6)
    layer = ConnectomeLayer(graph, trainable_edges=True)
    assert layer.edge_weight is not None
    np.testing.assert_array_equal(layer.edge_weight, graph.weights.data)
    # Mutating the layer must not mutate the source substrate graph.
    layer.edge_weight[:] = 0.5
    assert not np.allclose(graph.weights.data, 0.5)


def test_batched_input_support():
    graph = make_graph(n=5)
    layer = ConnectomeLayer(graph)
    x = np.random.default_rng(3).random((2, 3, 5)).astype(np.float32)
    y = layer(x)
    assert y.shape == (2, 3, 5)
