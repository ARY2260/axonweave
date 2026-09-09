# Biological Brain

One-sentence purpose: document `axonweave.load()`, the `BiologicalBrain` object, brain introspection, and neuron selection.

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
| `brain.substrate_id` | `str` | Substrate identifier, e.g. `"male-cns:v1.0"` |
| `brain.fingerprint` | `str` | SHA-256 identity hash of the loaded graph |
| `brain.annotations` | `Path` or `None` | Path to annotations Feather file, if installed |
| `brain.neurotransmitters` | `Path` or `None` | Path to neurotransmitter predictions, if installed |
| `brain.receptors` | `Path` or `None` | Path to receptor metadata, if present |

`BiologicalBrain` is explicitly a **computational wrapper**: the substrate identity and provenance live in the manifest on disk; the wrapper binds the graph to adapters.

## Introspection: brain.info()

`brain.info()` returns a `BrainInfo` snapshot of the loaded substrate:

```python
info = brain.info()

info.n_neurons        # 166700
info.n_edges          # 25600000 (approximate)
info.substrate_id     # "male-cns:v1.0"
info.fingerprint      # 64-char graph identity hash
info.has_annotations  # True / False per installed attachments
info.native_backend   # "rust" when the native core is available

print(info.summary())
# Substrate:   male-cns:v1.0
# Fingerprint: 3f9c1a2b7d4e5f60...
# Neurons:     166,700
# Connections: 25,600,000
# Annotations: yes  Neurotransmitters: yes  Receptors: no
# Native core: rust
```

`brain.capabilities()` returns the same facts as a machine-readable dict — backends, facades, dynamics and learning rules available on this instance — for use in experiment metadata and notebooks.

## The graph fingerprint

The fingerprint is a deterministic SHA-256 over the CSR structure (shape, body IDs, indptr, indices, data). Two loads of the same substrate version always produce the same fingerprint; any change in connectivity or body IDs changes it. It is used to:

- detect manifest identity mismatches at load time (`AXW002`)
- tag checkpoints so a trained model cannot be restored against an incompatible connectome (see [Checkpoints](checkpoints.md))

## Neuron selection

Selections resolve against **body IDs** — the stable biological identifiers preserved from the source dataset. Reach the selector through `brain.graph.neurons`:

```python
sel = brain.graph.neurons.all()             # every neuron
sel = brain.graph.neurons.ids([720575940632062920, 720575940610549233])
sel = brain.graph.neurons.by_mask(my_mask)  # boolean mask over graph order
```

Every selection is an ordered `NeuronSelection` with:

| Member | Meaning |
|---|---|
| `len(sel)` | Number of selected neurons |
| `sel.body_ids_list` | Selected body IDs, in request order |
| `sel.weights()` | Sparse sub-matrix restricted to the selection |
| `sel.mask()` | Boolean mask over all neurons |

Selection order is preserved exactly as requested, so `ids([a, b])` and `ids([b, a])` produce differently-ordered (but equivalent) selections. Unknown body IDs raise `AXW010` — the library never silently drops an ID.

:::DOC-NOTE
`by_type()` and `by_region()` resolve against selection tables built from the MaleCNS body-annotations file at install time (stored as `annotations.json` in the substrate directory). If the tables are missing — e.g. an older install — reinstalling the substrate builds them; until then these methods raise an actionable `AXW010` error rather than returning an empty selection.
:::

```python
sel = brain.graph.neurons.by_type("kenyon_cell")
sel = brain.graph.neurons.by_region("L")           # hemisphere-style region values
```

Annotation values are taken verbatim from the substrate's annotation columns (aliases such as `cell_type`/`type` and `side`/`region` are resolved by the builder). To discover the available values, query a known-missing one and read the error message, which lists up to 20 known values.

## Selections in layers

A selection can be handed directly to any backend's `ConnectomeLayer`. The layer then operates on the selected sub-network only — its input/output dimension equals the selection size, weights derive from the sub-matrix, and trainable edge parameters cover only the retained synapses:

```python
sel = brain.graph.neurons.by_type("kenyon_cell")

layer = brain.torch_layer(trainable_edges=True, selection=sel)  # torch.nn.Module over the sub-network
```

The layer records `selection_body_ids` so experiment metadata can always recover which neurons it computed over. Passing anything that is not a `NeuronSelection` raises `AXW010`.

## Framework adapters

```python
brain.torch_layer(trainable_edges=True, learnable_gain=True, selection=None)  # torch.nn.Module
brain.keras_layer(trainable_edges=True, selection=None)                       # tf.keras.layers.Layer
brain.layer(...)                                                              # alias for torch_layer
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
- [Checkpoints](checkpoints.md) — fingerprint-checked checkpoint identity.
