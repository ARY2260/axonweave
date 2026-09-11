from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ..core.graph import ConnectomeGraph
from ..errors import SchemaError


@dataclass
class GraphFingerprint:
    version: str
    content_hash: str
    n_neurons: int
    n_edges: int
    created: str
    schema_version: str = field(default="1.0")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "content_hash": self.content_hash,
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "created": self.created,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GraphFingerprint:
        return cls(
            version=d["version"],
            content_hash=d["content_hash"],
            n_neurons=d["n_neurons"],
            n_edges=d["n_edges"],
            created=d["created"],
            schema_version=d.get("schema_version", "1.0"),
        )

    def verify(self, graph: ConnectomeGraph) -> bool:
        current = compute_fingerprint(graph, schema_version=self.schema_version)
        return current.content_hash == self.content_hash


def _hash_graph(graph: ConnectomeGraph) -> str:
    from .. import native as _native

    return _native.csr_fingerprint(graph.weights, graph.body_ids)


def compute_fingerprint(graph: ConnectomeGraph, schema_version: str = "1.0") -> GraphFingerprint:
    return GraphFingerprint(
        version="1.0",
        content_hash=_hash_graph(graph),
        n_neurons=graph.n_neurons,
        n_edges=graph.n_edges,
        created=datetime.now(timezone.utc).isoformat(),
        schema_version=schema_version,
    )


def _noop_migration(graph: ConnectomeGraph) -> ConnectomeGraph:
    return graph


SCHEMA_VERSIONS: dict[str, Callable[[ConnectomeGraph], ConnectomeGraph]] = {
    "0.9": _noop_migration,
    "1.0": _noop_migration,
}


def migrate_graph(graph: ConnectomeGraph, from_version: str, to_version: str) -> ConnectomeGraph:
    if from_version == to_version:
        return graph
    if from_version not in SCHEMA_VERSIONS:
        raise SchemaError(f"AXW003: unknown schema version '{from_version}'")
    if to_version not in SCHEMA_VERSIONS:
        raise SchemaError(f"AXW003: unknown schema version '{to_version}'")
    if from_version == "0.9" and to_version == "1.0":
        return graph
    raise SchemaError(f"AXW003: no migration path from '{from_version}' to '{to_version}'")


def validate_schema(graph: ConnectomeGraph, expected_version: str) -> None:
    fp = compute_fingerprint(graph)
    if fp.schema_version != expected_version:
        raise SchemaError(
            f"AXW003: schema version mismatch; expected '{expected_version}', got '{fp.schema_version}'"
        )
