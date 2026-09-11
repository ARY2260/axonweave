from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from scipy import sparse

from ..core.graph import ConnectomeGraph
from .builder import ALIASES, _resolve


class DiskBackedGraphBuilder:
    def __init__(
        self,
        feather_path: str | Path,
        output_path: str | Path,
        batch_size: int = 100_000,
        tmp_dir: str | Path | None = None,
    ) -> None:
        self.feather_path = Path(feather_path)
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self._tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(
            prefix="axw_build_", dir=self.output_path.parent,
        ))
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._ids_path = self._tmp_dir / "axw_ids.npy"
        self._edges_path = self._tmp_dir / "axw_edges.npy"

    def _open_memmap(self, path: Path, dtype: np.dtype, shape: tuple[int]) -> np.memmap:
        return np.memmap(path, dtype=dtype, mode="w+", shape=shape)

    def _scan_ids(self) -> dict[int, int]:
        dataset = ds.dataset(self.feather_path, format="feather")
        names = set(dataset.schema.names)
        source_col = _resolve(names, ALIASES["source"], "source/body_pre column")
        target_col = _resolve(names, ALIASES["target"], "target/body_post column")

        ids = set()
        scanner = dataset.scanner(columns=[source_col, target_col], batch_size=self.batch_size)
        for batch in scanner.to_batches():
            ids.update(np.asarray(batch.column(0), dtype=np.int64).tolist())
            ids.update(np.asarray(batch.column(1), dtype=np.int64).tolist())

        body_ids = np.asarray(sorted(ids), dtype=np.int64)
        np.save(str(self._ids_path), body_ids)
        return {int(v): i for i, v in enumerate(body_ids)}

    def _stream_edges(self, lut: dict[int, int]) -> int:
        dataset = ds.dataset(self.feather_path, format="feather")
        names = set(dataset.schema.names)
        source_col = _resolve(names, ALIASES["source"], "source/body_pre column")
        target_col = _resolve(names, ALIASES["target"], "target/body_post column")
        weight_col = _resolve(names, ALIASES["weight"], "weight column")
        cols = [source_col, target_col, weight_col]

        n_edges = 0
        scanner = ds.dataset(self.feather_path, format="feather").scanner(
            columns=cols, batch_size=self.batch_size,
        )
        for batch in scanner.to_batches():
            n_edges += batch.num_rows

        if n_edges == 0:
            return 0

        edges_row = self._open_memmap(self._edges_path, np.int64, (n_edges,))
        edges_col = self._open_memmap(
            self._edges_path.with_suffix(".col.npy"), np.int64, (n_edges,),
        )
        edges_wgt = self._open_memmap(
            self._edges_path.with_suffix(".wgt.npy"), np.float32, (n_edges,),
        )

        offset = 0
        scanner = ds.dataset(self.feather_path, format="feather").scanner(
            columns=cols, batch_size=self.batch_size,
        )
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

        if n_edges == 0:
            body_ids = np.load(str(self._ids_path))
            n = len(body_ids)
            return sparse.csr_matrix((n, n), dtype=np.float32)

        body_ids = np.load(str(self._ids_path))
        n_neurons = len(body_ids)

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

    def build(self) -> ConnectomeGraph:
        lut = self._scan_ids()
        n_edges = self._stream_edges(lut)
        matrix = self._build_csr(n_edges)
        body_ids = np.load(str(self._ids_path))
        return ConnectomeGraph(matrix, body_ids)

    def cleanup(self) -> None:
        for p in self._tmp_dir.iterdir():
            p.unlink()
        self._tmp_dir.rmdir()
