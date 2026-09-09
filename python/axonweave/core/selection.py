"""Neuron selection (Phase 6 roadmap).

Selections resolve against body IDs, the stable biological identifiers of the
substrate. A ``NeuronSelection`` preserves the ordering it was constructed with
and can be applied to weight matrices, encoder wiring and readouts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import AxonWeaveError


@dataclass(frozen=True)
class NeuronSelection:
    """An ordered, validated set of neuron indices into a connectome graph."""

    graph: object
    indices: np.ndarray
    body_ids: np.ndarray

    def __len__(self) -> int:
        return int(len(self.indices))

    def __iter__(self):
        return iter(self.body_ids.tolist())

    @property
    def body_ids_list(self) -> list[int]:
        return self.body_ids.tolist()

    def weights(self) -> np.ndarray:
        """Sub-matrix of the connectome restricted to the selection."""
        m = self.graph.weights
        return m[self.indices][:, self.indices]

    def mask(self, n: int | None = None) -> np.ndarray:
        """Boolean mask of length ``n`` (default: all neurons), True on selection."""
        n = self.graph.n_neurons if n is None else n
        mask = np.zeros(n, dtype=bool)
        mask[self.indices] = True
        return mask


class NeuronSelector:
    """Entry point for selection queries; bound to a single graph."""

    def __init__(self, graph):
        self.graph = graph

    def _resolve(self, body_ids) -> NeuronSelection:
        ids = np.asarray(body_ids, dtype=np.int64)
        try:
            indices = np.asarray([self.graph.index(int(b)) for b in ids], dtype=np.int64)
        except KeyError:
            raise AxonWeaveError(
                "AXW010: selection contains body IDs not present in the substrate"
            ) from None
        return NeuronSelection(self.graph, indices, ids)

    def all(self) -> NeuronSelection:
        return NeuronSelection(
            self.graph,
            np.arange(self.graph.n_neurons, dtype=np.int64),
            np.asarray(self.graph.body_ids, dtype=np.int64),
        )

    def ids(self, body_ids) -> NeuronSelection:
        """Select by explicit body IDs; unknown IDs raise AXW010."""
        return self._resolve(list(body_ids))

    def by_mask(self, mask) -> NeuronSelection:
        """Select by boolean mask over graph order."""
        m = np.asarray(mask, dtype=bool)
        if m.shape != (self.graph.n_neurons,):
            raise ValueError(
                f"AXW010: mask length {m.shape[0]} != n_neurons {self.graph.n_neurons}")
        return NeuronSelection(
            self.graph,
            np.flatnonzero(m).astype(np.int64),
            np.asarray(self.graph.body_ids, dtype=np.int64)[np.flatnonzero(m)],
        )

    def by_type(self, neuron_type: str):
        """Select by cell type. Requires the annotations attachment."""
        return self._by_annotation("type", neuron_type)

    def by_region(self, region: str):
        """Select by anatomical region. Requires the annotations attachment."""
        return self._by_annotation("region", region)

    def _by_annotation(self, key: str, value: str):
        tables = getattr(self.graph, "selection_tables", None)
        if not tables or key not in tables:
            raise AxonWeaveError(
                f"AXW010: by_{key}() requires substrate selection tables; "
                "reinstall the substrate so annotations are built "
                "(axonweave substrate install male-cns:v1.0)"
            )
        table = tables[key]
        if value not in table:
            raise AxonWeaveError(
                f"AXW010: unknown {key} {value!r}; "
                f"known values: {sorted(table)[:20]}{' ...' if len(table) > 20 else ''}"
            )
        return self._resolve(table[value])
