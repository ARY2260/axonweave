# Selection & Sub-Network Examples

This page shows how to restrict computation to a biologically meaningful subset of the substrate: selecting neurons by ID, mask, or annotation, then running a `ConnectomeLayer` or a full PyTorch model on the sub-network.

```text
Full substrate (166,700 neurons)
        |
        v  brain.graph.neurons.by_type(...)
Selected sub-network (N neurons)
        |
        v  ConnectomeLayer(..., selection=sel)
Sparse propagation over retained synapses only
```

## Selecting neurons

Three selection entry points, all preserving the order you request:

```python
import axonweave

brain = axonweave.load("male-cns:v1.0")
sel_all = brain.graph.neurons.all()

# By body ID (order preserved exactly as given)
sel = brain.graph.neurons.ids([720575940632062920, 720575940610549233])

# By boolean mask over graph order
import numpy as np
mask = np.zeros(brain.n_neurons, dtype=bool)
mask[brain.graph.body_ids < 720575940600000000] = True
sel = brain.graph.neurons.by_mask(mask)

# By annotation (requires annotations built at substrate install time)
sel = brain.graph.neurons.by_type("kenyon_cell")
sel = brain.graph.neurons.by_region("L")
```

Unknown body IDs raise `AXW010` rather than being silently dropped; `by_type`/`by_region` errors list up to 20 known annotation values so you can discover the vocabulary.

:::DOC-NOTE
A selection is lightweight: it stores indices into the existing graph, never a copy of the connectome. `sel.weights()` materializes the sub-matrix on demand.
:::

## Inspecting a selection

```python
sel = brain.graph.neurons.by_type("kenyon_cell")

len(sel)             # number of neurons
sel.body_ids_list    # body IDs in request order
sel.mask()           # boolean mask over the full substrate
sub = sel.weights()  # sparse (N, N) sub-matrix
```

## Sub-network layer (NumPy reference)

The layer operates on the selected neuron space — input and output dimensions equal `len(sel)`:

```python
from axonweave.numpy import ConnectomeLayer

sel = brain.graph.neurons.ids([10, 20, 30, 40])
layer = ConnectomeLayer(brain.graph, selection=sel)

x = np.ones((2, 4), dtype=np.float32)
y = layer(x)          # shape (2, 4): propagation over retained synapses only
```

Equivalent to building the sub-graph by hand:

```python
from axonweave.numpy import ConnectomeLayer

manual = ConnectomeLayer(type("G", (), {"weights": sel.weights()})())
np.testing.assert_allclose(layer(x), manual(x), rtol=1e-6)
```

## Sub-network layer (PyTorch, trainable)

Trainable edges cover only the synapses retained in the selection — parameters outside the sub-network do not exist:

```python
import torch

sel = brain.graph.neurons.by_type("kenyon_cell")
layer = brain.torch_layer(trainable_edges=True, learnable_gain=True, selection=sel)

x = torch.randn(8, len(sel))
y = layer(x)
y.sum().backward()    # gradients flow to selected edges + gain only
```

The layer records which neurons it computed over:

```python
layer.selection_body_ids   # ndarray of body IDs, saved with checkpoints/metadata
```

## Sub-network inside a PyTorch model

Wrap the selected block with arbitrary native layers; dimensions line up at the selection boundary:

```python
import torch.nn as nn

sel = brain.graph.neurons.by_type("kenyon_cell")
n = len(sel)

model = nn.Sequential(
    nn.Linear(128, n),          # project onto the sub-network
    brain.torch_layer(selection=sel),
    nn.LayerNorm(n),
    nn.Linear(n, 4),            # read out from the sub-network
)
```

## Training a sub-network readout

Freeze the connectome and train only the interface around a selected population:

```python
sel = brain.graph.neurons.by_region("L")
n = len(sel)

model = nn.Sequential(
    nn.Linear(784, n),
    brain.torch_layer(selection=sel),          # frozen edges (default)
    nn.Linear(n, 10),
)

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad]
)
```

This is learning mode 1 (frozen connectome) from [Training](training.md), applied to a biologically motivated subset.

## What selections do NOT do

:::DOC-WARN
A selection restricts the **computational** sub-network. It does not claim the excluded neurons are biologically silent — in the fly brain, upstream neurons still influence the selected population. Selection is a modeling choice, not a biological statement about isolation.
:::

## Related

- [Biological Brain](brain.md) — selection API reference and introspection.
- [PyTorch Composition](examples-pytorch-composition.md) — full-brain layer composition.
- [Training](training.md) — the five learning modes.
