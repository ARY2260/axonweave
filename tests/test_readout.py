"""Tests for the first-class axonweave.readout package (Phase 6)."""
import numpy as np
import pytest

from axonweave.errors import ApiUsageError
from axonweave.readout import (
    ActionReadout,
    ClassificationHead,
    ClassificationReadout,
    RegressionReadout,
    TokenDecoder,
    TokenReadout,
)


def test_classification_logits_shape():
    r = ClassificationReadout(n_source=8, n_classes=3, seed=1)
    y = r(np.ones((2, 8), dtype=np.float32))
    assert y.shape == (2, 3)


def test_classification_accepts_selection_size():
    """Designed for neuron selections: n_source == len(sel)."""
    r = ClassificationReadout(n_source=4, n_classes=10, seed=2)
    y = r(np.ones(4, dtype=np.float32))
    assert y.shape == (10,)


def test_classification_dimension_guard():
    r = ClassificationReadout(n_source=8, n_classes=3)
    with pytest.raises(ApiUsageError, match="AXW010.*expected last dimension 8, got 5"):
        r(np.ones(5, dtype=np.float32))


def test_classification_deterministic_with_seed():
    a = ClassificationReadout(8, 3, seed=7)
    b = ClassificationReadout(8, 3, seed=7)
    x = np.arange(8, dtype=np.float32)
    np.testing.assert_allclose(a(x), b(x), rtol=1e-6)


def test_update_requires_trainable():
    r = ClassificationReadout(8, 3, trainable=False)
    with pytest.raises(ApiUsageError, match="AXW010.*not trainable"):
        r.update(np.ones((8, 3), dtype=np.float32))


def test_update_changes_weights_and_bias():
    r = ClassificationReadout(8, 3, trainable=True, seed=3)
    w0 = r.weight.copy()
    b0 = r.bias.copy()
    gw = np.full((8, 3), 0.1, dtype=np.float32)
    gb = np.full(3, 0.2, dtype=np.float32)
    r.update(gw, gb, lr=0.5)
    np.testing.assert_allclose(r.weight, w0 - 0.5 * gw, rtol=1e-6)
    np.testing.assert_allclose(r.bias, b0 - 0.5 * gb, rtol=1e-6)


def test_regression_readout():
    r = RegressionReadout(n_source=8, n_outputs=2, seed=4)
    y = r(np.ones((3, 8), dtype=np.float32))
    assert y.shape == (3, 2)


def test_readout_size_validation():
    with pytest.raises(ApiUsageError, match="AXW010.*positive"):
        ClassificationReadout(n_source=0, n_classes=3)
    with pytest.raises(ApiUsageError, match="AXW010.*positive"):
        RegressionReadout(n_source=4, n_outputs=-1)


def test_token_readout_guarded():
    t = TokenReadout(n_source=8, vocab_size=50, seed=5)
    logits = t(np.ones(8, dtype=np.float32))
    assert logits.shape == (50,)
    with pytest.raises(ApiUsageError, match="AXW010"):
        t(np.ones(6, dtype=np.float32))


def test_action_readout_guarded():
    a = ActionReadout(n_source=8, actions=3)
    assert a(np.ones(8, dtype=np.float32)) in (0, 1, 2)
    with pytest.raises(ApiUsageError, match="AXW010.*source neurons"):
        ActionReadout(n_source=2, actions=3)
    with pytest.raises(ApiUsageError, match="AXW010"):
        a(np.ones(4, dtype=np.float32))


def test_reexports_preserve_compat():
    """Legacy names remain importable from the unified namespace."""
    assert ClassificationHead is TokenDecoder


def test_readout_with_selection_end_to_end():
    """readout over a neuron selection sub-matrix activity."""
    from axonweave.core.brain import BiologicalBrain
    from tests.conftest import make_graph

    brain = BiologicalBrain(make_graph(8, 3))
    sel = brain.graph.neurons.ids([0, 10, 20])
    sub = sel.weights()
    x = np.ones(3, dtype=np.float32)
    activity = x @ sub
    r = ClassificationReadout(n_source=len(sel), n_classes=2, seed=1)
    logits = r(activity)
    assert logits.shape == (2,)
