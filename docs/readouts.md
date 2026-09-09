# Readouts

One-sentence purpose: document the `axonweave.readout` package — the task-facing interfaces that map neural activity onto classification, regression, token and action outputs.

The pipeline's output side has a single unified namespace:

```text
encoder  ->  brain  ->  readout
```

Encoders turn your data into neural input; readouts turn neural activity back into task output. Both are explicit, configurable interface choices — never implied by the substrate.

## Why a readout exists

The substrate produces neuron-sized activity (166,700 values for the full brain, or `len(sel)` for a [selection](brain.md)). No task consumes that directly. A readout selects, projects and shapes the activity into:

- class logits (classification)
- target values (regression)
- vocabulary logits (next-token prediction)
- motor actions (agent control)

## Available readouts

| Readout | Constructor | Output | Typical use |
|---|---|---|---|
| `ClassificationReadout` | `(n_source, n_classes, trainable=False, seed=0)` | class logits, shape `(..., n_classes)` | image/text classification |
| `RegressionReadout` | `(n_source, n_outputs=1, trainable=False, seed=0)` | values, shape `(..., n_outputs)` | scalar/vector targets |
| `TokenReadout` | `(n_source, vocab_size, seed=0)` | vocab logits, shape `(..., vocab_size)` | next-token prediction |
| `ActionReadout` | `(n_source, actions, continuous=False, seed=0)` | arg-max index (discrete) or clipped vector (continuous) | agent control |

All are importable as `axonweave.readout.X` and re-exported at top level (`axonweave.ClassificationReadout`, …).

## n_source: matching the neuron space

`n_source` is the number of neurons feeding the readout. It must match the last dimension of the activity you pass — mismatches raise `AXW010: expected last dimension N, got M` immediately.

Typical values:

```python
brain = axonweave.load("male-cns:v1.0")

# Full brain
r = axonweave.readout.ClassificationReadout(brain.n_neurons, n_classes=10)

# A neuron selection (preferred — see Biological Brain page)
sel = brain.graph.neurons.by_type("kenyon_cell")
r = axonweave.readout.ClassificationReadout(len(sel), n_classes=10)
```

## Reference-path training

`ClassificationReadout` and `RegressionReadout` support a NumPy-reference SGD step for the frozen-connectome mode (learning mode 1 — only interface parameters train):

```python
r = ClassificationReadout(n_source=len(sel), n_classes=2, trainable=True)
logits = r(activity)

# Host-framework users: use torch/tf optimizers on r.weight instead.
r.update(grad_weight, grad_bias, lr=1e-3)
```

:::DOC-NOTE
`update()` raises `AXW010` unless the readout was constructed with `trainable=True`. On the torch path, `BrainModel` wraps its own `nn.Linear` projections and training flows through the optimizer — these reference readouts are for the NumPy/agent path.
:::

Weights are deterministic per `seed`, so experiments reproduce exactly.

## Token and action readouts

`TokenReadout` and `ActionReadout` wrap the decoder implementations with explicit `n_source` guards:

```python
tok = TokenReadout(n_source=len(sel), vocab_size=50_000, seed=1)
logits = tok(activity)             # (..., 50_000) next-token logits

act = ActionReadout(n_source=len(sel), actions=4)
action = act(activity)             # arg-max action index
```

:::DOC-WARN
Next-token prediction is a computational task interface over the substrate. It does not imply the biological fly nervous system performs language modeling — see [Scientific Limitations](limitations.md) and [Tasks](tasks.md).
:::

## End-to-end with a selection

```python
import numpy as np

sel = brain.graph.neurons.ids([0, 10, 20])
r = axonweave.readout.ClassificationReadout(len(sel), n_classes=2, seed=1)

sub = sel.weights()                # (3, 3) sparse sub-network
activity = np.ones(3, dtype=np.float32) @ sub
logits = r(activity)               # shape (2,)
```

## Related

- [Biological Brain](brain.md) — neuron selections that feed `n_source`.
- [Tasks](tasks.md) — `brain.task(...)` with torch-side readout wiring.
- [Training](training.md) — learning modes; readouts are the trainable interface in mode 1.
- [Encoders & Decoders](encoders.md) — the input-side interfaces.
