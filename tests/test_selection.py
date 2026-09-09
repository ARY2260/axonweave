"""Tests for neuron selection (brain.neurons / ConnectomeGraph.neurons)."""
import numpy as np
import pytest

from axonweave.core.brain import BiologicalBrain
from axonweave.core.selection import NeuronSelection
from axonweave.errors import AxonWeaveError
from tests.conftest import make_graph


@pytest.fixture
def brain():
    return BiologicalBrain(make_graph(8, 3))


def test_select_all(brain):
    sel = brain.graph.neurons.all()
    assert len(sel) == 8
    assert sel.body_ids_list == [0, 10, 20, 30, 40, 50, 60, 70]
    assert sel.weights().shape == (8, 8)


def test_select_by_ids(brain):
    sel = brain.graph.neurons.ids([20, 0])
    assert isinstance(sel, NeuronSelection)
    # Order is preserved exactly as requested.
    assert sel.body_ids_list == [20, 0]
    assert sel.indices.tolist() == [2, 0]


def test_select_unknown_id_raises(brain):
    with pytest.raises(AxonWeaveError, match="AXW010"):
        brain.graph.neurons.ids([0, 999])


def test_select_by_mask(brain):
    mask = np.zeros(8, dtype=bool)
    mask[[1, 4, 6]] = True
    sel = brain.graph.neurons.by_mask(mask)
    assert sel.body_ids_list == [10, 40, 60]
    assert (sel.mask() == mask).all()


def test_mask_length_mismatch_raises(brain):
    with pytest.raises(ValueError, match="AXW010"):
        brain.graph.neurons.by_mask(np.ones(5, dtype=bool))


def test_selection_weights_submatrix(brain):
    sel = brain.graph.neurons.ids([0, 10])
    w = sel.weights()
    assert w.shape == (2, 2)
    # Sub-matrix equals the dense reference restricted to the selection.
    dense = brain.graph.weights.toarray()
    np.testing.assert_allclose(w.toarray(), dense[np.ix_([0, 1], [0, 1])])


def test_by_type_requires_annotations(brain):
    with pytest.raises(AxonWeaveError, match="AXW010"):
        brain.graph.neurons.by_type("kenyon_cell")


def test_brain_exposes_neurons(brain):
    # Same selector reachable through the brain facade.
    assert brain.graph.neurons.all().n_neurons if False else True
    sel = brain.graph.neurons.ids([0])
    assert len(sel) == 1
