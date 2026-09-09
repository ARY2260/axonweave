# Biological Brain

One-sentence purpose: document `axonweave.load()` and the `BiologicalBrain` object — the computational wrapper around a provisioned substrate.

:::DOC-WARN
`brain.neurons`, `brain.info()` and neuron-selection APIs are planned but not yet implemented. This page documents what exists today.
:::

## Loading

```python
import axonweave

brain = axonweave.load("male-cns:v1.0")
```

`load()` reads from the local substrate cache only. It never downloads data. If the substrate is missing it raises:

```text
AXW001: substrate 'male-cns:v1.0' is not installed.
Run `axonweave substrate install male-cns:v1.0`.
```

## What BiologicalBrain contains

| Attribute | Type | Meaning |
|---|---|---|
| `brain.graph` | `ConnectomeGraph` | Sparse CSR connectivity + body IDs |
| `brain.n_neurons` | `int` | Number of neurons (graph nodes) |
| `brain.annotations` | `Path` or `None` | Path to annotations Feather file, if installed |
| `brain.neurotransmitters` | `Path` or `None` | Path to neurotransmitter predictions, if installed |
| `brain.receptors` | `Path` or `None` | Path to receptor metadata, if present |

`BiologicalBrain` is explicitly a **computational wrapper**: the substrate identity and provenance live in the manifest on disk; the wrapper binds the graph to adapters.

## Framework adapters

```python
brain.torch_layer(trainable_edges=True, learnable_gain=True)   # torch.nn.Module
brain.keras_layer(trainable_edges=True)                        # tf.keras.layers.Layer
brain.layer(...)                                               # alias for torch_layer
```

These remain the stable low-level APIs. The high-level facades are additive:

```python
brain.task(input=..., output=..., dynamics="lif")   # supervised model (requires torch)
brain.agent(input=..., output=..., learning="stdp") # environment agent
brain.simulate(...)                                 # no-learning run
brain.experiment(...)                               # learning-enabled run
```

## Manifest identity

Every provisioned substrate carries a `manifest.json` with substrate ID, version, source URLs, per-file checksums, graph size and build status. `load()` validates identity (`AXW002` on mismatch) so you cannot silently compute against the wrong substrate.

## Related

- [Substrate Distribution](distribution.md) — how the substrate gets to disk.
- [Connectome](connectome.md) — the graph object in detail.
- [Framework API](framework.md) — tasks and agents.
