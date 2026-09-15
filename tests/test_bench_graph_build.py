"""Tests for the graph-construction benchmark suite (benchmarks/bench_graph_build.py).

The benchmark itself is a developer tool; these tests verify the properties it
relies on so a broken benchmark cannot silently report wrong numbers:

- fixture generation is deterministic (same seed -> same file) and correct;
- both build paths produce identical graphs and fingerprints at benchmark scale;
- the benchmark entrypoint runs end-to-end at reduced scale and reports
  fingerprint equality;
- JSONL output rows are well-formed and carry the required provenance fields.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCHMARKS_DIR))

import bench_graph_build as bench  # noqa: E402

from axonweave.core.brain import substrate_fingerprint  # noqa: E402
from axonweave.data.streaming_builder import DiskBackedGraphBuilder  # noqa: E402
from axonweave.data.builder import build_graph  # noqa: E402


def test_fixture_generation_is_deterministic(tmp_path):
    a = tmp_path / "a.feather"
    b = tmp_path / "b.feather"
    n1 = bench.generate_connectivity_feather(a, n_neurons=100, density=0.05, seed=7)
    n2 = bench.generate_connectivity_feather(b, n_neurons=100, density=0.05, seed=7)
    assert n1 == n2 == 495  # int(100 * 100 * 0.05)
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def test_both_build_paths_agree_at_benchmark_scale(tmp_path):
    feather_path = tmp_path / "conn.feather"
    bench.generate_connectivity_feather(feather_path, n_neurons=400, density=0.02, seed=3)
    g_mem = build_graph(feather_path, tmp_path / "mem.npz")
    with DiskBackedGraphBuilder(feather_path, tmp_path / "disk.npz", batch_size=2_000) as b:
        g_disk = b.build()
    assert g_mem.n_neurons == g_disk.n_neurons
    assert g_mem.n_edges == g_disk.n_edges
    assert substrate_fingerprint(g_mem) == substrate_fingerprint(g_disk)


def test_time_build_reports_fingerprint(tmp_path):
    feather_path = tmp_path / "conn.feather"
    bench.generate_connectivity_feather(feather_path, n_neurons=150, density=0.05, seed=11)
    row = bench.time_build("in-memory", lambda: build_graph(feather_path, tmp_path / "g.npz"))
    assert row["n_neurons"] == 150
    assert len(row["fingerprint"]) == 64
    assert row["seconds"] >= 0.0
    assert row["py_alloc_peak_bytes"] > 0


def test_benchmark_entrypoint_end_to_end(tmp_path, capsys):
    rc = bench.main([
        "--n-neurons", "300",
        "--density", "0.02",
        "--label", "pytest",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fp equal" in out
    assert "True" in out


def test_benchmark_jsonl_rows_are_well_formed(tmp_path, capsys, monkeypatch):
    # Copy the benchmark module into tmp_path so its JSONL output (written next
    # to the script file) lands inside the test's temp directory.
    (tmp_path / "bench_graph_build.py").write_text(
        (BENCHMARKS_DIR / "bench_graph_build.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    import importlib
    import sys as _sys

    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("bench_graph_build")
    importlib.reload(mod)

    rc = mod.main([
        "--n-neurons", "200",
        "--density", "0.03",
        "--jsonl",
        "--label", "jsonl-test",
    ])
    assert rc == 0
    results = tmp_path / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["label"] == "jsonl-test"
        assert row["native_active"] in (True, False)
        assert row["fingerprint_equal"] is True
        assert set(row["params"]) == {"n_neurons", "density", "batch_size", "n_edges"}
        assert len(row["fingerprint"]) == 64
    _sys.modules.pop("bench_graph_build", None)


def test_peak_rss_helper_returns_int():
    assert isinstance(bench.peak_rss_bytes(), int)
    assert bench.peak_rss_bytes() >= 0


def test_benchmark_rejects_fingerprint_mismatch(tmp_path, capsys, monkeypatch):
    """If the two builders ever disagree, the benchmark must fail loudly."""
    feather_path = tmp_path / "conn.feather"
    bench.generate_connectivity_feather(feather_path, n_neurons=100, density=0.05, seed=5)

    real_time_build = bench.time_build

    def skewed_build(builder_name, fn):
        row = real_time_build(builder_name, fn)
        if builder_name == "disk-backed":
            row["fingerprint"] = "0" * 64
        return row

    monkeypatch.setattr(bench, "time_build", skewed_build)
    rc = bench.main(["--n-neurons", "100", "--density", "0.05"])
    assert rc == 1
    assert "correctness bug" in capsys.readouterr().out
