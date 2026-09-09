"""Tests for neuron selections wired into ConnectomeLayers and annotation
selection tables (by_type/by_region against built annotation columns)."""
import json

import numpy as np
import pytest

from axonweave.core.brain import BiologicalBrain
from axonweave.numpy import ConnectomeLayer as NumpyLayer
from tests.conftest import make_graph


@pytest.fixture
def brain():
    return BiologicalBrain(make_graph(8, 3))


def test_numpy_layer_on_selection(brain):
    sel = brain.graph.neurons.ids([0, 10, 20, 30])
    layer = NumpyLayer(brain.graph, selection=sel)
    assert layer.graph_weights.shape == (4, 4)
    x = np.random.default_rng(0).random((2, 4)).astype(np.float32)
    y = layer(x)
    assert y.shape == (2, 4)
    # Matches propagation on the equivalent hand-built sub-graph.
    ref = NumpyLayer(type("G", (), {"weights": sel.weights()})())
    np.testing.assert_allclose(y, ref(x), rtol=1e-6)


def test_numpy_layer_selection_preserves_order(brain):
    a = brain.graph.neurons.ids([0, 10])
    b = brain.graph.neurons.ids([10, 0])
    la = NumpyLayer(brain.graph, selection=a)
    lb = NumpyLayer(brain.graph, selection=b)
    x2 = np.array([[1.0, 2.0]], dtype=np.float32)
    # Same neurons in reverse order: feeding the permuted input yields the
    # permuted output — selection order defines the layer's neuron space.
    np.testing.assert_allclose(la(x2)[0], lb(x2[:, ::-1])[0, ::-1], rtol=1e-6)
    assert la.selection_body_ids.tolist() == [0, 10]
    assert lb.selection_body_ids.tolist() == [10, 0]


def test_layer_rejects_non_selection(brain):
    with pytest.raises(ValueError, match="AXW010"):
        NumpyLayer(brain.graph, selection=[0, 1, 2])


def test_selection_layer_weights_are_trainable_subset(brain):
    sel = brain.graph.neurons.by_mask(np.arange(8) % 2 == 0)
    layer = NumpyLayer(brain.graph, trainable_edges=True, selection=sel)
    assert layer.edge_weight is not None
    assert len(layer.edge_weight) == sel.weights().nnz
    assert layer.selection_body_ids.tolist() == [0, 20, 40, 60]


def test_annotation_selection_tables(brain, tmp_path):
    """by_type/by_region resolve against the built annotations.json tables."""
    brain.graph.selection_tables = {
        "type": {"kenyon_cell": [0, 10], "optic": [20]},
        "region": {"brain": [0, 10, 20]},
    }
    sel = brain.graph.neurons.by_type("kenyon_cell")
    assert sel.body_ids_list == [0, 10]
    sel = brain.graph.neurons.by_region("brain")
    assert len(sel) == 3
    with pytest.raises(Exception, match="AXW010"):
        brain.graph.neurons.by_type("mushroom_body")


def test_build_annotations_from_feather(tmp_path):
    """build_annotations resolves aliased columns into selection tables."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.feather as feather

    from axonweave.data.builder import build_annotations

    table = pa.table({
        "bodyId": [10, 20, 30],
        "cell_type": ["kenyon_cell", "kenyon_cell", None],
        "side": ["L", "R", "L"],
    })
    p = tmp_path / "annotations.feather"
    feather.write_feather(table, p)
    tables = build_annotations(p, tmp_path / "annotations.json")
    assert tables["type"]["kenyon_cell"] == [10, 20]
    assert tables["region"]["L"] == [10, 30]
    on_disk = json.loads((tmp_path / "annotations.json").read_text())
    assert on_disk == tables


def test_builder_still_roundtrips(tmp_path):
    """Existing build_graph behavior unchanged."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.feather as feather

    from axonweave.data.builder import build_graph

    table = pa.table({
        "body_pre": [1, 2],
        "body_post": [2, 1],
        "weight": [3.0, 4.0],
    })
    p = tmp_path / "conn.feather"
    feather.write_feather(table, p)
    g = build_graph(p, tmp_path / "graph.npz")
    assert g.n_neurons == 2
    assert g.n_edges == 2
