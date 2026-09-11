"""Tests for typed neuron metadata API (NeuronMetadata / NeuronMetadataStore)."""
import json

import numpy as np
import pytest

from axonweave.core.brain import BiologicalBrain
from axonweave.core.metadata import NeuronMetadata, NeuronMetadataStore
from tests.conftest import make_graph


ROWS = {
    "10": {"cell_type": "kenyon_cell", "region": "alpha_lobe", "hemisphere": "left"},
    "20": {"cell_type": "kenyon_cell", "region": "alpha_lobe", "hemisphere": "right"},
    "30": {"cell_type": "projection", "region": "antennal_lobe", "hemisphere": "left"},
    "40": {"cell_type": "projection", "region": "antennal_lobe", "hemisphere": "right"},
}


@pytest.fixture
def json_store(tmp_path):
    p = tmp_path / "annotations.json"
    p.write_text(json.dumps(ROWS), encoding="utf-8")
    return NeuronMetadataStore(p)


@pytest.fixture
def feather_store(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.feather as feather

    p = tmp_path / "annotations.feather"
    table = pa.table(
        {
            "body_id": pa.array([10, 20, 30, 40], type=pa.int64()),
            "cell_type": ["kenyon_cell", "kenyon_cell", "projection", "projection"],
            "region": ["alpha_lobe", "alpha_lobe", "antennal_lobe", "antennal_lobe"],
            "hemisphere": ["left", "right", "left", "right"],
        }
    )
    feather.write_feather(table, p)
    return NeuronMetadataStore(p)


@pytest.fixture
def store(request, json_store, feather_store):
    if request.param == "json":
        return json_store
    return feather_store


@pytest.mark.parametrize("store", ["json", "feather"], indirect=True)
def test_get_single(store):
    meta = store.get(20)
    assert isinstance(meta, NeuronMetadata)
    assert meta.body_id == 20
    assert meta.cell_type == "kenyon_cell"
    assert meta.region == "alpha_lobe"
    assert meta.hemisphere == "right"


@pytest.mark.parametrize("store", ["json", "feather"], indirect=True)
def test_get_unknown_returns_none(store):
    assert store.get(999) is None


@pytest.mark.parametrize("store", ["json", "feather"], indirect=True)
def test_get_batch(store):
    metas = store.get_batch(np.array([40, 10, 999], dtype=np.int64))
    assert [m.body_id for m in metas] == [40, 10]


@pytest.mark.parametrize("store", ["json", "feather"], indirect=True)
def test_query_filters(store):
    kenyon = store.query(cell_type="kenyon_cell")
    np.testing.assert_array_equal(kenyon, np.array([10, 20], dtype=np.int64))
    left = store.query(hemisphere="left")
    np.testing.assert_array_equal(left, np.array([10, 30], dtype=np.int64))
    combined = store.query(cell_type="projection", hemisphere="right")
    np.testing.assert_array_equal(combined, np.array([40], dtype=np.int64))
    assert store.query(cell_type="does-not-exist").shape == (0,)


@pytest.mark.parametrize("store", ["json", "feather"], indirect=True)
def test_enumeration_and_flags(store):
    assert store.types() == ["kenyon_cell", "projection"]
    assert store.regions() == ["alpha_lobe", "antennal_lobe"]
    assert store.has_types
    assert store.has_regions
    assert store.has_hemispheres


@pytest.mark.parametrize("store", ["json", "feather"], indirect=True)
def test_summary(store):
    summary = store.summary()
    assert "4 neurons" in summary
    assert "kenyon_cell" in summary


def test_empty_store_degrades_gracefully():
    store = NeuronMetadataStore()
    assert store.get(123) is None
    assert store.get_batch(np.array([1, 2, 3], dtype=np.int64)) == []
    assert store.query().size == 0
    assert store.types() == []
    assert store.regions() == []
    assert not store.has_types
    assert not store.has_regions
    assert not store.has_hemispheres
    assert "no annotations" in store.summary()


def test_store_missing_path_degrades(tmp_path):
    store = NeuronMetadataStore(tmp_path / "does-not-exist.feather")
    assert store.get(10) is None
    assert not store.has_types


def test_brain_metadata_lazy_load(tmp_path):
    p = tmp_path / "annotations.json"
    p.write_text(json.dumps(ROWS), encoding="utf-8")
    brain = BiologicalBrain(make_graph(8, 3), annotations=p)
    store = brain.metadata
    assert isinstance(store, NeuronMetadataStore)
    assert brain.metadata is store
    assert brain.metadata.get(10).cell_type == "kenyon_cell"


def test_brain_metadata_without_annotations():
    brain = BiologicalBrain(make_graph(8, 3))
    assert brain.metadata.summary()
    assert brain.metadata.get(10) is None