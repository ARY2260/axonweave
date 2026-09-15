from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from scipy import sparse

from ..core.graph import ConnectomeGraph
from .builder import ALIASES, _resolve


class DiskBackedGraphBuilder:
    """Stream a connectivity Feather file into a CSR graph using O(edges-per-batch)
    RAM instead of holding every edge in memory.

    Body IDs are collected in a first scan pass (Python set — the only
    unbounded structure), then edge rows/columns/weights are appended to
    on-disk memmaps batch-by-batch and finally reduced to CSR via the native
    core's ``build_csr_from_coo``.

    The resulting artifact matches :func:`axonweave.data.builder.build_graph`
    exactly: ``graph.npz`` plus a ``.json`` sidecar recording the canonical
    content fingerprint (``native.csr_fingerprint``), neuron count and edge
    count.
    """

    def __init__(
        self,
        feather_path: str | Path,
        output_path: str | Path,
        batch_size: int = 100_000,
        tmp_dir: str | Path | None = None,
    ) -> None:
        self.feather_path = Path(feather_path)
        self.output_path = Path(output_path)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if tmp_dir:
            self._tmp_dir = Path(tmp_dir)
            self._owns_tmp = False
        else:
            self._tmp_dir = Path(tempfile.mkdtemp(prefix="axw_build_"))
            self._owns_tmp = True
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._ids_path = self._tmp_dir / "axw_ids.npy"
        self._edges_path = self._tmp_dir / "axw_edges.npy"

    # -- column resolution ---------------------------------------------------

    def _columns(self, dataset) -> tuple[str, str, str]:
        names = set(dataset.schema.names)
        source_col = _resolve(names, ALIASES["source"], "source/body_pre column")
        target_col = _resolve(names, ALIASES["target"], "target/body_post column")
        weight_col = _resolve(names, ALIASES["weight"], "weight column")
        return source_col, target_col, weight_col

    # -- passes ---------------------------------------------------------------

    def _scan_ids(self) -> dict[int, int]:
        dataset = ds.dataset(self.feather_path, format="feather")
        source_col, target_col, _ = self._columns(dataset)

        ids: set[int] = set()
        scanner = dataset.scanner(columns=[source_col, target_col], batch_size=self.batch_size)
        for batch in scanner.to_batches():
            ids.update(np.asarray(batch.column(0), dtype=np.int64).tolist())
            ids.update(np.asarray(batch.column(1), dtype=np.int64).tolist())

        body_ids = np.asarray(sorted(ids), dtype=np.int64)
        np.save(str(self._ids_path), body_ids)
        return {int(v): i for i, v in enumerate(body_ids)}

    def _stream_edges(self, lut: dict[int, int]) -> int:
        dataset = ds.dataset(self.feather_path, format="feather")
        cols = list(self._columns(dataset))

        n_edges = 0
        scanner = dataset.scanner(columns=cols, batch_size=self.batch_size)
        for batch in scanner.to_batches():
            n_edges += batch.num_rows

        if n_edges == 0:
            return 0

        edges_row = np.memmap(self._edges_path, dtype=np.int64, mode="w+", shape=(n_edges,))
        edges_col = np.memmap(
            self._edges_path.with_suffix(".col.npy"), dtype=np.int64, mode="w+", shape=(n_edges,),
        )
        edges_wgt = np.memmap(
            self._edges_path.with_suffix(".wgt.npy"), dtype=np.float32, mode="w+", shape=(n_edges,),
        )

        offset = 0
        scanner = dataset.scanner(columns=cols, batch_size=self.batch_size)
        for batch in scanner.to_batches():
            src = np.asarray(batch.column(0), dtype=np.int64)
            dst = np.asarray(batch.column(1), dtype=np.int64)
            wgt = np.asarray(batch.column(2), dtype=np.float32)
            n = len(src)
            edges_row[offset:offset + n] = np.fromiter(
                (lut[int(x)] for x in src), dtype=np.int64, count=n,
            )
            edges_col[offset:offset + n] = np.fromiter(
                (lut[int(x)] for x in dst), dtype=np.int64, count=n,
            )
            edges_wgt[offset:offset + n] = wgt
            offset += n

        edges_row.flush()
        edges_col.flush()
        edges_wgt.flush()
        return n_edges

    def _build_csr(self, n_edges: int) -> sparse.csr_matrix:
        from .. import native as _native

        body_ids = np.load(str(self._ids_path))
        n_neurons = len(body_ids)
        if n_edges == 0:
            return sparse.csr_matrix((n_neurons, n_neurons), dtype=np.float32)

        edges_row = np.memmap(self._edges_path, dtype=np.int64, mode="r", shape=(n_edges,))
        edges_col = np.memmap(
            self._edges_path.with_suffix(".col.npy"), dtype=np.int64, mode="r", shape=(n_edges,),
        )
        edges_wgt = np.memmap(
            self._edges_path.with_suffix(".wgt.npy"), dtype=np.float32, mode="r", shape=(n_edges,),
        )

        return _native.build_csr_from_coo(
            edges_row, edges_col, edges_wgt, (n_neurons, n_neurons),
        )

    # -- public API -----------------------------------------------------------

    def build(self) -> ConnectomeGraph:
        """Stream-build the graph, persist ``graph.npz`` + fingerprint JSON,
        and return the :class:`ConnectomeGraph`."""
        lut = self._scan_ids()
        n_edges = self._stream_edges(lut)
        matrix = self._build_csr(n_edges)
        body_ids = np.load(str(self._ids_path))
        graph = ConnectomeGraph(matrix, body_ids)

        from .. import native as _native

        fingerprint = _native.csr_fingerprint(graph.weights, graph.body_ids)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        graph.save(self.output_path)
        self.output_path.with_suffix(".json").write_text(
            json.dumps({
                "graph_fingerprint": fingerprint,
                "n_neurons": graph.n_neurons,
                "n_edges": graph.n_edges,
                "builder": "disk-backed",
            }, indent=2),
            encoding="utf-8",
        )
        return graph

    def cleanup(self) -> None:
        """Remove scratch files; the temp directory only when we created it."""
        for p in sorted(self._tmp_dir.glob("axw_edges*")):
            p.unlink(missing_ok=True)
        self._ids_path.unlink(missing_ok=True)
        if self._owns_tmp:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def __enter__(self) -> "DiskBackedGraphBuilder":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()
