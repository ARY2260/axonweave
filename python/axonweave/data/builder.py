from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pyarrow.dataset as ds
from scipy import sparse
from .. import native as _native
from ..core.graph import ConnectomeGraph

ALIASES = {
    "source": ("body_pre", "pre", "source", "bodyId_pre", "body_pre_id"),
    "target": ("body_post", "post", "target", "bodyId_post", "body_post_id"),
    "weight": ("weight", "weights", "synapse_count", "count"),
}

# Annotation column aliases for neuron selection (by_type / by_region).
# The MaleCNS body-annotations file publishes cell-type and side/region
# information under these names; resolved with the same tolerant scheme as
# the connectivity columns.
ANNOTATION_ALIASES = {
    "body": ("body_id", "bodyId", "body", "id"),
    "type": ("cell_type", "type", "neuron_type", "class"),
    "region": ("side", "region", "super_class", "roi", "hemisphere"),
}

def _resolve(names, candidates, label):
    for c in candidates:
        if c in names:
            return c
    from ..errors import SchemaError
    raise SchemaError(f"AXW003: could not resolve {label}; available columns={sorted(names)}")

def build_graph(feather_path, output_path, batch_size=100_000):
    """Build a sparse graph from a Feather dataset.

    The builder streams Arrow record batches but currently retains edge arrays before
    CSR construction. Large-production deployments should use the planned disk-backed
    Rust builder to reduce peak memory.
    """
    dataset = ds.dataset(feather_path, format="feather")
    names = set(dataset.schema.names)
    source_col = _resolve(names, ALIASES["source"], "source/body_pre column")
    target_col = _resolve(names, ALIASES["target"], "target/body_post column")
    weight_col = _resolve(names, ALIASES["weight"], "weight column")
    cols = [source_col, target_col, weight_col]

    ids = set()
    scanner = dataset.scanner(columns=cols[:2], batch_size=batch_size)
    for batch in scanner.to_batches():
        ids.update(np.asarray(batch.column(0), dtype=np.int64).tolist())
        ids.update(np.asarray(batch.column(1), dtype=np.int64).tolist())
    body_ids = np.asarray(sorted(ids), dtype=np.int64)
    lut = {int(v): i for i, v in enumerate(body_ids)}

    rows_parts, cols_parts, weight_parts = [], [], []
    scanner = dataset.scanner(columns=cols, batch_size=batch_size)
    for batch in scanner.to_batches():
        src = np.asarray(batch.column(0), dtype=np.int64)
        dst = np.asarray(batch.column(1), dtype=np.int64)
        weight = np.asarray(batch.column(2), dtype=np.float32)
        rows_parts.append(np.fromiter((lut[int(x)] for x in src), dtype=np.int64, count=len(src)))
        cols_parts.append(np.fromiter((lut[int(x)] for x in dst), dtype=np.int64, count=len(dst)))
        weight_parts.append(weight)

    rows = np.concatenate(rows_parts) if rows_parts else np.empty(0, dtype=np.int64)
    cols_ = np.concatenate(cols_parts) if cols_parts else np.empty(0, dtype=np.int64)
    weights = np.concatenate(weight_parts) if weight_parts else np.empty(0, dtype=np.float32)
    matrix = sparse.coo_matrix((weights, (rows, cols_)), shape=(len(body_ids), len(body_ids)), dtype=np.float32).tocsr()
    graph = ConnectomeGraph(matrix, body_ids)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    graph.save(output_path)
    fingerprint = _native.csr_fingerprint(matrix, body_ids)
    Path(output_path).with_suffix('.json').write_text(json.dumps({"graph_fingerprint": fingerprint, "n_neurons": graph.n_neurons, "n_edges": graph.n_edges}, indent=2), encoding="utf-8")
    return graph


def build_annotations(feather_path, output_path, batch_size=100_000):
    """Extract selection tables (type / region -> body IDs) from the
    annotations Feather file and write them as ``annotations.json``.

    Missing columns degrade to absent keys rather than failing the install —
    type/region selection then raises AXW010 with a clear message at query
    time, matching the selection API contract.
    """
    import pyarrow.dataset as arrow_ds

    dataset = arrow_ds.dataset(feather_path, format="feather")
    names = set(dataset.schema.names)
    resolved = {}
    for key, candidates in ANNOTATION_ALIASES.items():
        for c in candidates:
            if c in names:
                resolved[key] = c
                break
    tables: dict[str, dict[str, list[int]]] = {}
    if "body" in resolved:
        body_col = resolved["body"]
        scanner = dataset.scanner(
            columns=[body_col] + [v for k, v in resolved.items() if k != "body"],
            batch_size=batch_size,
        )
        for batch in scanner.to_batches():
            bodies = np.asarray(batch.column(0))
            for i, key in enumerate([k for k in resolved if k != "body"], start=1):
                values = np.asarray(batch.column(i))
                table = tables.setdefault(key, {})
                for b, v in zip(bodies.tolist(), values.tolist()):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    table.setdefault(str(v), []).append(int(b))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(tables, indent=1), encoding="utf-8")
    return tables
