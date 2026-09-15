"""Unit, functional and integration tests for the PyTorch ConnectomeLayer."""
import pytest

from tests.conftest import make_graph

torch = pytest.importorskip("torch")
from axonweave.torch import ConnectomeLayer  # noqa: E402
from axonweave.errors import UnsupportedDeviceError  # noqa: E402


@pytest.fixture
def small_graph():
    return make_graph(n=8, seed=11)


def test_forward_matches_dense_reference(small_graph):
    """Functional: sparse propagation equals dense reference matmul (y = x @ W)."""
    layer = ConnectomeLayer(small_graph)
    x = torch.randn(4, small_graph.n_neurons, dtype=torch.float32)
    y = layer(x)
    expected = x @ torch.tensor(small_graph.weights.toarray())
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)


def test_output_shape(small_graph):
    layer = ConnectomeLayer(small_graph)
    y = layer(torch.randn(2, small_graph.n_neurons))
    assert y.shape == (2, small_graph.n_neurons)


def test_trainable_edges_receive_gradients(small_graph):
    """Functional: trainable edge weights participate in autograd."""
    layer = ConnectomeLayer(small_graph, trainable_edges=True)
    assert isinstance(layer.edge_weight, torch.nn.Parameter)
    x = torch.randn(2, small_graph.n_neurons)
    layer(x).sum().backward()
    assert layer.edge_weight.grad is not None
    assert torch.isfinite(layer.edge_weight.grad).all()


def test_frozen_edges_have_no_grad(small_graph):
    layer = ConnectomeLayer(small_graph, trainable_edges=False)
    assert not isinstance(layer.edge_weight, torch.nn.Parameter)
    x = torch.randn(2, small_graph.n_neurons)
    y = layer(x)
    assert not y.requires_grad


def test_learnable_gain_updates(small_graph):
    layer = ConnectomeLayer(small_graph, learnable_gain=True)
    x = torch.randn(2, small_graph.n_neurons)
    layer(x).sum().backward()
    assert layer.gain.grad is not None and float(layer.gain.grad) != 0.0


def test_bias_added(small_graph):
    layer = ConnectomeLayer(small_graph, bias=True)
    x = torch.zeros(2, small_graph.n_neurons)
    y = layer(x)
    # Zero input through sparse matmul leaves zeros; bias must remain.
    assert torch.count_nonzero(y) == small_graph.n_neurons


def test_wrong_feature_dimension_raises(small_graph):
    layer = ConnectomeLayer(small_graph)
    with pytest.raises(ValueError, match="AXW010"):
        layer(torch.randn(2, small_graph.n_neurons + 1))


def test_unsupported_device_raises_actionable_error(small_graph):
    """Integration: AXW004 on a device string the backend rejects."""
    with pytest.raises(UnsupportedDeviceError, match="AXW004"):
        ConnectomeLayer(small_graph, device="xpu:99")


def test_training_step_updates_weights(small_graph):
    """Integration: one optimizer step changes trainable parameters."""
    layer = ConnectomeLayer(small_graph, trainable_edges=True, learnable_gain=True)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    before = layer.edge_weight.detach().clone()
    x = torch.randn(4, small_graph.n_neurons)
    loss = layer(x).pow(2).mean()
    loss.backward()
    opt.step()
    assert not torch.equal(layer.edge_weight, before)


def test_batched_3d_input(small_graph):
    layer = ConnectomeLayer(small_graph)
    x = torch.randn(2, 3, small_graph.n_neurons)
    assert layer(x).shape == (2, 3, small_graph.n_neurons)


# ---------------------------------------------------------------------------
# API improvements: AXW010 consistency, repr, dtype, signal_policy guard
# ---------------------------------------------------------------------------

def test_bad_selection_raises_api_usage_error(small_graph):
    """Selection type errors raise ApiUsageError (AXW010), not plain ValueError."""
    from axonweave.errors import ApiUsageError

    with pytest.raises(ApiUsageError, match="AXW010"):
        ConnectomeLayer(small_graph, selection=[1, 2, 3])


def test_signal_policy_warns_axw007(small_graph):
    """Passing signal_policy warns (AXW007) instead of being silently ignored."""
    with pytest.warns(UserWarning, match="AXW007"):
        layer = ConnectomeLayer(small_graph, signal_policy=object())
    assert layer.signal_policy is not None


def test_extra_repr_reports_structure(small_graph):
    layer = ConnectomeLayer(
        small_graph, trainable_edges=True, learnable_gain=True, bias=True)
    r = repr(layer)
    assert "n_neurons=8" in r
    assert "trainable_edges=True" in r
    assert "learnable_gain=True" in r
    assert "bias=True" in r


def test_float64_input_preserves_dtype(small_graph):
    """Double-precision inputs round-trip through the sparse path."""
    layer = ConnectomeLayer(small_graph)
    x = torch.randn(2, small_graph.n_neurons, dtype=torch.float64)
    y = layer(x)
    assert y.dtype == torch.float64
    expected = x @ torch.tensor(small_graph.weights.toarray(), dtype=torch.float64)
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)


def test_graph_weights_exposed(small_graph):
    """The backing CSR is reachable (used by ConnectomeBlock dynamics)."""
    layer = ConnectomeLayer(small_graph)
    assert layer.graph_weights is small_graph.weights
