"""Tests for the substrate data pipeline: checksums, builder, registry, installer."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest
from scipy import sparse

from axonweave import ConnectomeGraph, SubstrateRegistry
from axonweave.data.builder import build_graph
from axonweave.data.checksums import (
    UPSTREAM_CHECKSUMS,
    expected_checksum,
    verify_checksum,
)
from axonweave.data.installer import _sha256, install_male_cns
from axonweave.data.manifest import MALE_CNS
from axonweave.errors import DatasetIntegrityError, SchemaError, SubstrateNotInstalledError


# ---------------------------------------------------------------------------
# Checksum verification (unit)
# ---------------------------------------------------------------------------

def test_expected_checksum_declared_or_none():
    """Every MaleCNS file has a real upstream GCS md5 or an explicit None."""
    for key in MALE_CNS["files"]:
        entry = expected_checksum(MALE_CNS["id"], key)
        md5 = entry.get("md5_base64")
        sha = entry.get("sha256")
        assert md5 is None or isinstance(md5, str)
        assert sha is None or (isinstance(sha, str) and len(sha) == 64)
    # The connectivity file (the one actually ingested) must have a hash.
    assert expected_checksum(MALE_CNS["id"], "connectivity").get("md5_base64")


def test_verify_checksum_accepts_match():
    digest = hashlib.sha256(b"x").hexdigest()
    verify_checksum(digest, {"sha256": digest}, "connectivity")  # must not raise
    verify_checksum("irrelevant", {"md5_base64": "abc=="}, "connectivity", "abc==")


def test_verify_checksum_accepts_unpublished():
    """No upstream hash declared -> skip explicitly, no error."""
    verify_checksum("deadbeef", {}, "connectivity")


def test_verify_checksum_rejects_mismatch():
    with pytest.raises(DatasetIntegrityError, match="AXW002"):
        verify_checksum("a" * 64, {"sha256": "b" * 64}, "connectivity")
    with pytest.raises(DatasetIntegrityError, match="AXW002"):
        verify_checksum("x", {"md5_base64": "good=="}, "connectivity", "bad==")


def test_md5_base64_of_file_matches_gcs_format(tmp_path):
    import base64
    from axonweave.data.checksums import md5_base64_of_file

    p = tmp_path / "f.bin"
    p.write_bytes(b"axonweave")
    expected = base64.b64encode(hashlib.md5(b"axonweave").digest()).decode()
    assert md5_base64_of_file(p) == expected


# ---------------------------------------------------------------------------
# Test fixtures: synthetic feather connectivity file
# ---------------------------------------------------------------------------

def _write_connectivity_feather(path: Path, n_edges=20, seed=5):
    rng = np.random.default_rng(seed)
    src = rng.integers(100, 140, n_edges)
    dst = rng.integers(100, 140, n_edges)
    weight = rng.integers(1, 10, n_edges).astype(np.float32)
    table = pa.table({
        "body_pre": pa.array(src, pa.int64()),
        "body_post": pa.array(dst, pa.int64()),
        "weight": pa.array(weight, pa.float32()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(table, path)
    return path


# ---------------------------------------------------------------------------
# Builder (unit + functional)
# ---------------------------------------------------------------------------

def test_build_graph_creates_graph_and_fingerprint(tmp_path):
    feather_path = _write_connectivity_feather(tmp_path / "conn.feather")
    out = tmp_path / "sub" / "graph.npz"
    graph = build_graph(feather_path, out)

    assert out.exists()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["n_neurons"] == graph.n_neurons
    assert meta["n_edges"] == graph.n_edges
    assert len(meta["graph_fingerprint"]) == 64
    assert graph.n_edges > 0


def test_build_graph_deterministic_fingerprint(tmp_path):
    """Functional: identical input -> identical fingerprint."""
    feather_path = _write_connectivity_feather(tmp_path / "conn.feather")
    out1 = tmp_path / "g1.npz"
    out2 = tmp_path / "g2.npz"
    build_graph(feather_path, out1)
    build_graph(feather_path, out2)
    f1 = json.loads(out1.with_suffix(".json").read_text())["graph_fingerprint"]
    f2 = json.loads(out2.with_suffix(".json").read_text())["graph_fingerprint"]
    assert f1 == f2


def test_build_graph_preserves_body_ids(tmp_path):
    feather_path = _write_connectivity_feather(tmp_path / "conn.feather", seed=9)
    graph = build_graph(feather_path, tmp_path / "g.npz")
    original = pa.feather.read_table(feather_path)
    ids = set(original["body_pre"].to_pylist()) | set(original["body_post"].to_pylist())
    assert set(graph.body_ids.tolist()) == ids


def test_build_graph_weight_column_aliases(tmp_path):
    """Schema resolution accepts upstream column-name variants (AXW003 otherwise)."""
    rng = np.random.default_rng(11)
    table = pa.table({
        "bodyId_pre": pa.array(rng.integers(1, 5, 6), pa.int64()),
        "bodyId_post": pa.array(rng.integers(1, 5, 6), pa.int64()),
        "synapse_count": pa.array(rng.integers(1, 4, 6).astype(np.float32)),
    })
    p = tmp_path / "alias.feather"
    feather.write_feather(table, p)
    graph = build_graph(p, tmp_path / "g.npz")
    assert graph.n_neurons > 0


def test_build_graph_unresolvable_schema_raises(tmp_path):
    table = pa.table({"a": pa.array([1], pa.int64())})
    p = tmp_path / "bad.feather"
    feather.write_feather(table, p)
    with pytest.raises(SchemaError, match="AXW003"):
        build_graph(p, tmp_path / "g.npz")


# ---------------------------------------------------------------------------
# Registry (unit + functional)
# ---------------------------------------------------------------------------

def _fake_installed_registry(root: Path) -> SubstrateRegistry:
    reg = SubstrateRegistry(root)
    target = reg.path("male-cns:v1.0")
    target.mkdir(parents=True, exist_ok=True)
    g = ConnectomeGraph(sparse.eye(4, dtype=np.float32, format="csr"), np.arange(4) * 5)
    g.save(target / "graph.npz")
    (target / "manifest.json").write_text(json.dumps({
        "id": "male-cns:v1.0", "status": "installed", "version": "v1.0",
    }), encoding="utf-8")
    return reg


def test_registry_load_installed(tmp_path):
    reg = _fake_installed_registry(tmp_path)
    brain = reg.load("male-cns:v1.0")
    assert brain.n_neurons == 4


def test_registry_load_missing_raises(tmp_path):
    with pytest.raises(SubstrateNotInstalledError, match="AXW001"):
        SubstrateRegistry(tmp_path).load("male-cns:v1.0")


def test_registry_manifest_identity_mismatch_raises(tmp_path):
    reg = _fake_installed_registry(tmp_path)
    manifest_path = reg.path("male-cns:v1.0") / "manifest.json"
    meta = json.loads(manifest_path.read_text())
    meta["id"] = "other-substrate:v9"
    manifest_path.write_text(json.dumps(meta))
    with pytest.raises(DatasetIntegrityError, match="AXW002"):
        reg.load("male-cns:v1.0")


def test_registry_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("AXONWEAVE_HOME", str(tmp_path / "custom"))
    assert SubstrateRegistry().root == tmp_path / "custom"


# ---------------------------------------------------------------------------
# Installer (integration, network mocked)
# ---------------------------------------------------------------------------

class _FakeHeaders:
    """Server hashes keyed by requested filename, mirroring GCS behavior."""

    per_file = {}

    def __init__(self, filename):
        self._md5 = _FakeHeaders.per_file.get(filename, "")

    def get(self, key, default=None):
        if key == "x-goog-hash" and self._md5:
            return f"md5={self._md5}"
        return default


class _FakeResponse:
    """Minimal streamed response for the installer's download path."""

    def __init__(self, payload: bytes, filename: str = "", status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.headers = _FakeHeaders(filename)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self._payload


# Pristine server-reported hashes (what GCS would serve), captured before
# any test mutates the registry.
_SERVER_MD5 = {
    f: expected_checksum(MALE_CNS["id"], key)["md5_base64"]
    for key, f in MALE_CNS["files"].items()
    if expected_checksum(MALE_CNS["id"], key).get("md5_base64")
}


def _install_with_fake_downloads(tmp_path, payload=b"fake-feather-bytes"):
    # Serve the real upstream md5 per file so the happy path verifies cleanly.
    _FakeHeaders.per_file = dict(_SERVER_MD5)
    calls: list[str] = []

    def fake_get(url, headers=None, stream=True, timeout=None):
        calls.append(url)
        return _FakeResponse(payload, filename=url.rsplit("/", 1)[-1])

    def fake_build_graph(feather_path, output_path, batch_size=100_000):
        """Write a tiny real graph instead of parsing the fake feather bytes."""
        g = ConnectomeGraph(sparse.eye(3, dtype=np.float32, format="csr"), np.arange(3) * 7)
        g.save(output_path)
        return g

    with mock.patch("axonweave.data.installer.requests.Session.get", side_effect=fake_get), \
         mock.patch("axonweave.data.installer.build_graph", side_effect=fake_build_graph), \
         mock.patch("shutil.copy2"):
        brain = install_male_cns(root=tmp_path)

    return brain, calls


def test_installer_downloads_required_files_and_installs(tmp_path):
    brain, calls = _install_with_fake_downloads(tmp_path)
    assert brain.n_neurons == 3
    required = ["connectivity", "annotations", "neurotransmitters"]
    for key in required:
        filename = MALE_CNS["files"][key]
        assert any(c.endswith(filename) for c in calls)
    manifest = json.loads((tmp_path / "substrates" / "male-cns-v1.0" / "manifest.json").read_text())
    assert manifest["status"] == "installed"
    assert manifest["id"] == "male-cns:v1.0"
    for key in required:
        assert len(manifest["files"][key]["sha256"]) == 64


def test_installer_optional_groups(tmp_path):
    _, calls = _install_with_fake_downloads(tmp_path)
    stats_file = MALE_CNS["files"]["stats"]
    assert not any(c.endswith(stats_file) for c in calls)


def test_installer_checksum_mismatch_aborts(tmp_path):
    """Integration: a corrupted download with a declared hash must fail AXW002."""
    UPSTREAM_CHECKSUMS[MALE_CNS["id"]]["annotations"]["md5_base64"] = "0" * 20
    try:
        with pytest.raises(DatasetIntegrityError, match="AXW002"):
            _install_with_fake_downloads(tmp_path)
        # No manifest written: substrate not activated.
        assert not (tmp_path / "substrates" / "male-cns-v1.0" / "manifest.json").exists()
    finally:
        UPSTREAM_CHECKSUMS[MALE_CNS["id"]]["annotations"]["md5_base64"] = (
            "UKdxh3DFciDxYLpPQxq4ng=="
        )


def test_sha256_of_known_bytes(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"axonweave")
    assert _sha256(p) == hashlib.sha256(b"axonweave").hexdigest()
