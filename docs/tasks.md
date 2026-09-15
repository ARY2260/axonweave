# Tasks


:::DOC-WARN
The task API requires PyTorch (`pip install "axonweave[torch]"`) and is **experimental**. Keras and JAX task adapters are planned.
:::

## The pipeline

```text
data
  ↓
encoder      (application data → neural currents)
  ↓
brain        (sparse connectome + dynamics)
  ↓
readout      (neural activity → task output)
  ↓
loss         (host framework)
```

## Creating a task

```python
import axonweave

brain = axonweave.load("male-cns:v1.0")

model = brain.task(
    input=axonweave.encoders.ImageEncoder(shape=(28, 28, 1), n_target=1024),
    output=axonweave.decoders.ClassificationHead(10),
    dynamics="rate",          # "lif", "adaptive_lif", "rate"
    trainable_edges=False,    # Mode 1: frozen connectome
)
```

Under the hood this constructs a `BrainModel` with `Input`/`Readout` interfaces sized from the encoder/decoder.

## Training

```python
history = model.fit(train_loader, epochs=3, lr=1e-3)
```

`fit` uses the host framework's optimizer (Adam by default) and cross-entropy loss. Nothing trains unless something can: with a frozen substrate and no connected interfaces, `fit` raises `AXW010` instead of silently doing nothing.

## Training modes

See [Learning & Plasticity](learning.md) for the five modes. The task API exposes:

- `trainable_edges=False` — frozen substrate (Mode 1)
- `trainable_edges=True` — synaptic weights on existing edges (Mode 2)
- `learning="stdp"` / `"dopamine_stdp"` — local plasticity alongside backprop (Modes 4/5)

## Task families

| Family | Status |
|---|---|
| Classification | Experimental |
| Next-token prediction (via TokenEncoder/TokenDecoder) | Experimental |
| Regression (via continuous readouts) | Planned |
| Reinforcement learning (via agent API) | Experimental |
| Custom tasks (compose your own modules) | Supported |

## Scientific framing

A next-token experiment means: token IDs → encoder → connectome dynamics → readout optimized against cross-entropy. It does **not** mean the fly brain understands language. The task interface is yours; the substrate remains source data. See [Scientific Reference](scientific-reference.md).

## Related

- [Framework API](framework.md) — BrainModel and ConnectomeBlock details.
- [Encoders & Decoders](encoders.md).
- [Learning & Plasticity](learning.md).
