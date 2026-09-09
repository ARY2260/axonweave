# Connectome

One-sentence purpose: explain the graph representation — nodes, edges, weights, sparsity and identity — and the APIs to inspect it.

## Representation

The connectome is a **directed weighted graph** stored as a SciPy CSR sparse matrix:

- **Nodes** — neurons, identified by their original upstream **body IDs** (int64). AxonWeave never renumbers them: index 0 in the matrix maps to a stable body ID through `brain.graph.body_ids`.
- **Edges** — directed synaptic connections `pre → post`.
- **Edge weights** — derived from upstream synaptic counts. These are source data, preserved as loaded.
- **Sparsity** — the full MaleCNS graph is ~166,700 neurons × ~25.6M directed edges. A dense 166,700² matrix would need ~100 GB; AxonWeave never materializes it.

## ConnectomeGraph API

```python
graph = brain.graph

graph.n_neurons        # node count
graph.n_edges          # edge count (CSR nonzeros)
graph.body_ids         # int64 array: matrix index -> upstream body ID
graph.index(body_id)   # body ID -> matrix index (KeyError if unknown)
graph.neighbors(body_id)  # (target_body_ids, weights) for outgoing edges

graph.save(path)       # compressed .npz round-trip
graph = ConnectomeGraph.load(path)
```

## Raw graph vs propagation graph

The **raw connectome** is the structural source: connectivity as published. The **propagation graph** is the same matrix used computationally by layers and dynamics. AxonWeave keeps them identical by default; trainable modes derive computational weights as `W = W₀ + ΔW` on existing edges only, preserving topology.

## Identity and fingerprints

Graph construction writes a SHA-256 fingerprint over the CSR data, indices, indptr and body IDs. Fingerprints are stored in the substrate manifest and are the basis for future checkpoint compatibility checks.

## Why not dense?

- Memory: dense full-connectome matrices are infeasible (~100 GB for MaleCNS).
- Semantics: the biological graph is sparse; dense representation would invent zero-weight synapses.
- Performance: sparse CSR × dense batch matmul is the hot path and is well-supported by NumPy/SciPy, PyTorch and TensorFlow.

## Related

- [Biological Brain](brain.md)
- [Backends](backends.md) — sparse operator support per framework.
- [Scientific Reference](scientific-reference.md) — data provenance.
