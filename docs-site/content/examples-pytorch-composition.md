# PyTorch Composition Example

This example shows the intended composition boundary: ordinary PyTorch layers before and after the connectome layer.

```python
import torch.nn as nn
from axonweave import load

brain = load("male-cns:v1.0")
connectome = brain.torch_layer(trainable_edges=True, learnable_gain=True)

model = nn.Sequential(
    nn.Linear(128, brain.n_neurons),
    connectome,
    nn.LayerNorm(brain.n_neurons),
    nn.Linear(brain.n_neurons, 4),
)
```

For very large full-brain states, evaluate memory and latency before selecting this architecture.
