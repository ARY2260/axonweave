# Getting Started

From a clean environment to a working substrate: install the package, provision the MaleCNS connectome, load a brain and wire it into PyTorch or Keras.

## Requirements

Python 3.10–3.14 is targeted. The core path requires NumPy, SciPy, PyArrow and the substrate cache dependencies. Framework integrations are optional extras.

## Install

```bash
pip install axonweave
```

PyTorch:

```bash
pip install "axonweave[torch]"
```

TensorFlow/Keras:

```bash
pip install "axonweave[tensorflow]"
```

All currently supported optional integrations:

```bash
pip install "axonweave[all]"
```

## Provision the biological substrate

```bash
axonweave substrate install male-cns:v1.0
```

The command owns download, validation, sparse graph construction and local activation. You do not need to manually download Feather files.

:::DOC-WARN
The full source release is multi-gigabyte. Provisioning is intentionally separate from `pip install` so installing Python dependencies does not unexpectedly download biological data.
:::

## Load

```python
import axonweave

brain = axonweave.load("male-cns:v1.0")
print(brain.n_neurons)
```

## PyTorch

```python
import torch
from axonweave.torch import ConnectomeLayer

layer = ConnectomeLayer(
    brain.graph,
    trainable_edges=True,
    learnable_gain=True,
)

x = torch.randn(2, brain.n_neurons)
y = layer(x)
```

## Keras

```python
from axonweave.keras import ConnectomeLayer

layer = ConnectomeLayer(
    brain.graph,
    trainable_edges=True,
    learnable_gain=True,
)
```

## Constraints

A full-connectome state is large. Production applications should deliberately select input projection, internal state, readout and batching strategies. Do not assume a full dense tensor of all neurons is cheap simply because the graph itself is sparse.
