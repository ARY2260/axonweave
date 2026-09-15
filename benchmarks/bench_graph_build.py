"""Graph-construction benchmark suite (PLAN.md Phase 4: Profiling/benchmark suite).

Benchmarks the three graph-build paths available today and reports wall time,
peak RSS and content fingerprint equality:

1. ``in-memory``   — ``axonweave.data.builder.build_graph`` (batches held in RAM).
2. ``disk-backed`` — ``axonweave.data.streaming_builder.DiskBackedGraphBuilder``
                     (Python streaming; O(batch) RAM).
3. ``native kernels`` — same disk-backed build, but reporting whether the
                     compiled ``axonweave._native`` CSR reduction is active
                     (the Rust streaming builder — native Arrow/IPC ingestion,
                     parallel construction — is the Phase 4 roadmap item).

Each run writes its result to stdout as a table and appends a JSON line to
``benchmarks/results.jsonl`` when ``--jsonl`` is passed, so benchmark reports
(Phase 7) can accumulate machine-readable history.

Usage::

    python benchmarks/bench_graph_build.py                       # default sizes
    python benchmarks/bench_graph_build.py --n-neurons 5000 --density 0.001
    python benchmarks/bench_graph_build.py --jsonl --label nightly
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from axonweave import native as N
from axonweave.data.builder import build_graph
from axonweave.data.streaming_builder import DiskBackedGraphBuilder


def generate_connectivity_feather(path: Path, n_neurons: int, density: float, seed: int = 42) -> int:
    """Write a synthetic connectivity Feather file; return the edge count."""
    rng = np.random.default_rng(seed)
    n_edges = max(1, int(n_neurons * (n_neurons - 1) * density))
    src = rng.integers(0, n_neurons, n_edges)
    dst = rng.integers(0, n_neurons, n_edges)
    w = rng.integers(1, 10, n_edges).astype(np.float32)
    table = pa.table({
        "body_pre": pa.array(src, pa.int64()),
        "body_post": pa.array(dst, pa.int64()),
        "weight": pa.array(w, pa.float32()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(table, path)
    return n_edges


def peak_rss_bytes() -> int:
    """Best-effort peak-RSS in bytes across platforms (0 when unavailable)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # KiB -> B (Linux/macOS)
    except ImportError:
        try:
            import psutil
            return int(psutil.Process().memory_info().peak_wset)  # Windows
        except ImportError:
            return 0


def time_build(builder_name: str, fn) -> dict:
    tracemalloc.start()
    t0 = time.perf_counter()
    graph = fn()
    elapsed = time.perf_counter() - t0
    _py_peak, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "builder": builder_name,
        "seconds": round(elapsed, 4),
        "py_alloc_peak_bytes": py_peak,
        "peak_rss_bytes": peak_rss_bytes(),
        "n_neurons": graph.n_neurons,
        "n_edges": graph.n_edges,
        "fingerprint": N.csr_fingerprint(graph.weights, graph.body_ids),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bench_graph_build")
    p.add_argument("--n-neurons", type=int, default=2000)
    p.add_argument("--density", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=100_000)
    p.add_argument("--label", default="", help="free-form label recorded in the JSONL row")
    p.add_argument("--jsonl", action="store_true", help="append rows to benchmarks/results.jsonl")
    args = p.parse_args(argv)

    workdir = Path(__file__).parent / "bench_graph_build_work"
    feather_path = workdir / "connectivity.feather"
    print(f"Generating fixture: n_neurons={args.n_neurons:,} density={args.density} ...")
    n_edges = generate_connectivity_feather(feather_path, args.n_neurons, args.density)
    print(f"  edges: {n_edges:,}")

    rows = []
    rows.append(time_build("in-memory", lambda: build_graph(feather_path, workdir / "graph_mem.npz")))
    with DiskBackedGraphBuilder(feather_path, workdir / "graph_disk.npz", batch_size=args.batch_size) as b:
        rows.append(time_build("disk-backed", b.build))

    fingerprints = {r["fingerprint"] for r in rows}
    fingerprint_equal = len(fingerprints) == 1
    for r in rows:
        r["fingerprint_equal"] = fingerprint_equal
        r["native_active"] = N._HAS_NATIVE
        r["label"] = args.label
        r["timestamp"] = datetime.now(timezone.utc).isoformat()
        r["params"] = {
            "n_neurons": args.n_neurons, "density": args.density,
            "batch_size": args.batch_size, "n_edges": n_edges,
        }
        r["python"] = platform.python_version()

    name_w = max(len(r["builder"]) for r in rows) + 2
    print(f"\n{'builder':<{name_w}} {'seconds':>10} {'py peak':>12} {'rss peak':>12} {'fp equal':>9}")
    for r in rows:
        print(f"{r['builder']:<{name_w}} {r['seconds']:>10.4f} "
              f"{r['py_alloc_peak_bytes']:>12,} {r['peak_rss_bytes']:>12,} "
              f"{str(r['fingerprint_equal']):>9}")
    if not fingerprint_equal:
        print("ERROR: fingerprints differ across builders -- this is a correctness bug", flush=True)
        return 1

    if args.jsonl:
        out = Path(__file__).parent / "results.jsonl"
        with out.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"\nAppended {len(rows)} rows to {out}")

    if N._HAS_NATIVE:
        print("\nnative core: ACTIVE (compiled CSR reduction used in the disk-backed path)")
        print("Rust streaming builder (native Arrow ingestion): Phase 4 roadmap, not yet implemented.")
    else:
        print("\nnative core: not compiled -- disk-backed path used the NumPy/SciPy reference kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
