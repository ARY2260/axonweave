from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ApiUsageError
from ..core.graph import ConnectomeGraph


@dataclass
class MixedPrecisionPolicy:
    dtype: str = "float32"
    loss_scale: float = 1.0
    dynamic_loss_scaling: bool = True
    initial_loss_scale: float = 2**16
    growth_interval: int = 2000
    growth_factor: float = 2.0
    backoff_factor: float = 0.5

    _SUPPORTED_DTYPES = frozenset({"float16", "bfloat16", "float32"})

    def __post_init__(self):
        if self.dtype not in self._SUPPORTED_DTYPES:
            raise ApiUsageError(
                f"AXW010: unsupported dtype {self.dtype!r}; "
                f"expected one of {sorted(self._SUPPORTED_DTYPES)}"
            )
        if self.loss_scale <= 0:
            raise ApiUsageError(
                f"AXW010: loss_scale must be positive, got {self.loss_scale}"
            )
        if self.initial_loss_scale <= 0:
            raise ApiUsageError(
                f"AXW010: initial_loss_scale must be positive, got {self.initial_loss_scale}"
            )
        if self.growth_interval < 1:
            raise ApiUsageError(
                f"AXW010: growth_interval must be >= 1, got {self.growth_interval}"
            )
        if self.growth_factor <= 0:
            raise ApiUsageError(
                f"AXW010: growth_factor must be positive, got {self.growth_factor}"
            )
        if self.backoff_factor <= 0 or self.backoff_factor >= 1:
            raise ApiUsageError(
                f"AXW010: backoff_factor must be in (0, 1), got {self.backoff_factor}"
            )

    @property
    def is_mixed(self) -> bool:
        return self.dtype != "float32"

    def effective_dtype(self) -> str:
        return self.dtype

    def static_scale(self) -> float | None:
        if self.dynamic_loss_scaling:
            return None
        return self.loss_scale


class DistributedPartitioner:
    """Partition a ConnectomeGraph into roughly equal sub-graphs.

    Uses a greedy balanced partitioning algorithm based on successive
    approximation of the Fiduccia-Mattheyses heuristic. The goal is
    balanced vertex counts with minimal edge cuts.
    """

    def assign_partitions(self, graph: ConnectomeGraph, n_partitions: int) -> np.ndarray:
        if n_partitions < 1:
            raise ApiUsageError(
                f"AXW010: n_partitions must be >= 1, got {n_partitions}"
            )
        if n_partitions == 1:
            return np.zeros(graph.n_neurons, dtype=np.int64)

        n = graph.n_neurons
        if n_partitions >= n:
            return np.arange(n, dtype=np.int64)

        assignment = self._greedy_partition(graph, n_partitions)
        return assignment

    def partition(self, graph: ConnectomeGraph, n_partitions: int) -> list[ConnectomeGraph]:
        assignment = self.assign_partitions(graph, n_partitions)
        m = graph.weights.tocoo()

        subgraphs: list[ConnectomeGraph] = []
        for p in range(n_partitions):
            mask = assignment == p
            if not np.any(mask):
                continue
            local_ids = np.where(mask)[0]
            local_lookup = np.full(graph.n_neurons, -1, dtype=np.int64)
            local_lookup[local_ids] = np.arange(len(local_ids), dtype=np.int64)

            rows = m.row
            cols = m.col
            data = m.data
            keep = mask[rows] & mask[cols]
            new_rows = local_lookup[rows[keep]]
            new_cols = local_lookup[cols[keep]]
            new_data = data[keep]
            size = len(local_ids)
            sub_matrix = (
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
            ) if size == 0 else (new_data.astype(np.float32), new_cols.astype(np.int32), new_rows.astype(np.int32))
            from scipy import sparse
            sm = sparse.csr_matrix(
                (sub_matrix[0], (sub_matrix[1], sub_matrix[2])),
                shape=(size, size),
            )
            subgraphs.append(ConnectomeGraph(sm, graph.body_ids[local_ids].copy()))

        return subgraphs

    def _greedy_partition(self, graph: ConnectomeGraph, n_partitions: int) -> np.ndarray:
        n = graph.n_neurons
        target_size = n / n_partitions
        m = graph.weights.tocsr()

        assignment = np.full(n, -1, dtype=np.int64)
        partition_sizes = np.zeros(n_partitions, dtype=np.int64)

        degrees = np.array(m.getnnz(axis=1).ravel(), dtype=np.float64).ravel()
        if degrees.sum() == 0:
            degrees = np.ones(n, dtype=np.float64)

        order = np.argsort(-degrees)

        for node in order:
            best_partition = -1
            best_gain = -np.inf

            neighbors = m.indices[m.indptr[node]:m.indptr[node + 1]]
            neighbor_weights = m.data[m.indptr[node]:m.indptr[node + 1]]

            for p in range(n_partitions):
                if partition_sizes[p] >= target_size and partition_sizes[np.argmin(partition_sizes)] < target_size:
                    continue

                gain = 0.0
                for nb, w in zip(neighbors, neighbor_weights):
                    if assignment[nb] == p:
                        gain += w
                    elif assignment[nb] >= 0:
                        gain -= w

                balance_penalty = abs(partition_sizes[p] + 1 - target_size)
                score = gain - 0.01 * balance_penalty

                if score > best_gain:
                    best_gain = score
                    best_partition = p

            if best_partition < 0:
                best_partition = int(np.argmin(partition_sizes))

            assignment[node] = best_partition
            partition_sizes[best_partition] += 1

        for _ in range(3):
            self._kahura_swap(graph, assignment, partition_sizes, n_partitions)

        return assignment

    def _kahura_swap(self, graph: ConnectomeGraph, assignment: np.ndarray,
                     partition_sizes: np.ndarray, n_partitions: int) -> None:
        m = graph.weights.tocsr()
        n = graph.n_neurons
        improved = True

        while improved:
            improved = False
            for node in np.random.permutation(n):
                old_part = assignment[node]
                neighbors = m.indices[m.indptr[node]:m.indptr[node + 1]]
                neighbor_weights = m.data[m.indptr[node]:m.indptr[node + 1]]

                edge_gains = np.zeros(n_partitions, dtype=np.float64)
                for nb, w in zip(neighbors, neighbor_weights):
                    p = assignment[nb]
                    if p >= 0:
                        edge_gains[p] += w

                best_part = old_part
                best_score = edge_gains[old_part]

                for p in range(n_partitions):
                    if p == old_part:
                        continue
                    score = edge_gains[p]
                    balance_diff = abs(partition_sizes[p] + 1 - graph.n_neurons / n_partitions)
                    balance_old = abs(partition_sizes[old_part] - 1 - graph.n_neurons / n_partitions)
                    if score > best_score or (score == best_score and balance_diff < balance_old):
                        best_score = score
                        best_part = p

                if best_part != old_part:
                    partition_sizes[old_part] -= 1
                    partition_sizes[best_part] += 1
                    assignment[node] = best_part
                    improved = True
