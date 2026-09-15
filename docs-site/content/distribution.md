# Substrate Distribution

The Python wheel and the biological substrate are separate artifacts.

## Online

```bash
pip install axonweave
axonweave substrate install male-cns:v1.0
```

## Memory-bounded install

The default install assembles the graph in memory. On constrained machines, use the disk-backed builder, which streams edge batches to scratch files and never holds the full edge list in RAM:

```bash
axonweave substrate install male-cns:v1.0 --disk-backed
```

or programmatically:

```python
from axonweave.data import DiskBackedGraphBuilder

with DiskBackedGraphBuilder("connectivity.feather", "graph.npz", batch_size=100_000) as b:
    graph = b.build()
```

Both paths produce an identical graph and an identical content fingerprint, recorded in the substrate manifest and re-verified every time the substrate is loaded.

## Integrity

Each install records a SHA-256 content fingerprint of the built graph (canonical CSR payload + body IDs) in the manifest. `axonweave substrate verify` and every `SubstrateRegistry.load()` re-derive this fingerprint from the stored artifact and refuse a mismatched graph with `AXW002`, so a corrupted or swapped `graph.npz` cannot be silently activated.

## Cache

The default cache is under the platform's user cache directory and can be redirected with `AXONWEAVE_HOME`.

## Offline deployment

A future `.awb` pack format is planned. The intended workflow is:

```bash
axonweave substrate pack male-cns:v1.0 --output male-cns-v1.0.awb
```

then on an isolated machine:

```bash
axonweave substrate install ./male-cns-v1.0.awb
```

## Why not bundle it in PyPI?

The upstream release includes files ranging from tens of megabytes to multi-gigabyte synapse resources. A normal Python wheel should contain software, not force every installation to transfer the entire biological archive.
