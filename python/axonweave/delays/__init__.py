from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from ..errors import ApiUsageError


@dataclass(frozen=True)
class UniformDelay:
    min_ms: float
    max_ms: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.min_ms, self.max_ms, size=n).astype(np.float32)


@dataclass(frozen=True)
class FixedDelay:
    value_ms: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(n, self.value_ms, dtype=np.float32)


@dataclass(frozen=True)
class NormalDelay:
    mean_ms: float
    std_ms: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raw = rng.normal(self.mean_ms, self.std_ms, size=n)
        return np.clip(raw, 0.0, None).astype(np.float32)


@dataclass
class SynapticDelayEngine:
    max_delay_ms: float
    dt: float = 1.0
    n_neurons: int = 0
    delay_distribution: object = field(default_factory=lambda: UniformDelay(0.0, 10.0))
    seed: int | None = None

    def __post_init__(self):
        from .. import native as _native

        self._rng = np.random.default_rng(self.seed)
        self._ring: object | None = None
        self._edge_indices: np.ndarray | None = None
        self._pre_indices: np.ndarray | None = None
        self._post_indices: np.ndarray | None = None
        self._delay_ticks: np.ndarray | None = None
        self._edge_weights: np.ndarray | None = None
        self._n_delay_slots = 0
        self._initialized = False
        self._native = _native

    def _slots(self) -> int:
        return max(1, int(np.ceil(self.max_delay_ms / self.dt)) + 1)

    def init(self, n_neurons: int) -> None:
        self.n_neurons = n_neurons
        self._n_delay_slots = self._slots()
        self._ring = self._native.create_delay_ring(self._n_delay_slots, n_neurons)
        self._initialized = True

    def get_delays(self, graph, delay_distribution=None) -> np.ndarray:
        n_edges = graph.n_edges
        dist = delay_distribution or self.delay_distribution
        coo = graph.weights.tocoo()
        self._edge_indices = np.arange(n_edges)
        self._pre_indices = coo.row.astype(np.int64)
        self._post_indices = coo.col.astype(np.int64)
        self._edge_weights = coo.data.astype(np.float32)
        delays_ms = dist.sample(n_edges, self._rng)
        self._delay_ticks = np.maximum(
            np.round(delays_ms / self.dt).astype(np.int64), 0
        )
        n = graph.n_neurons
        self.n_neurons = n
        self._n_delay_slots = self._slots()
        self._ring = self._native.create_delay_ring(self._n_delay_slots, n)
        self._initialized = True
        return delays_ms

    def step(
        self, spike_train: np.ndarray, pre_activity: np.ndarray, dt: float
    ) -> np.ndarray:
        if self._ring is None or not self._initialized:
            raise ApiUsageError(
                "AXW010: SynapticDelayEngine not initialized; "
                "call init(n_neurons) or get_delays(graph) first."
            )
        pre_activity = np.asarray(pre_activity, dtype=np.float32)
        if self._pre_indices is None or self._delay_ticks is None:
            return np.zeros(self.n_neurons, dtype=np.float32)
        return self._ring.step(
            pre_activity,
            self._pre_indices,
            self._post_indices,
            self._delay_ticks,
            self._edge_weights,
        )

    def apply_delays(
        self, W: sparse.spmatrix, delays: np.ndarray, pre_activity: np.ndarray
    ) -> np.ndarray:
        if self._ring is None or not self._initialized:
            raise ApiUsageError(
                "AXW010: SynapticDelayEngine not initialized; "
                "call init(n_neurons) or get_delays(graph) first."
            )
        coo = W.tocoo()
        pre_idx = coo.row.astype(np.int64)
        post_idx = coo.col.astype(np.int64)
        weights = coo.data.astype(np.float32)

        delay_ticks = np.maximum(np.round(delays / self.dt).astype(np.int64), 0)
        return self._ring.read(pre_idx, post_idx, delay_ticks, weights)
