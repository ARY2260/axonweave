# Framework API

AxonWeave exposes the connectome as a **computational substrate**: you compose it with your own encoders, decoders, dynamics and learning rules. The substrate stays fixed unless you explicitly make it trainable.

## BrainModel: supervised tasks

For classification or token-prediction style tasks, wrap the substrate in a `BrainModel`:

```python
import torch
import axonweave
from axonweave.frameworks.torch import BrainModel, Input, Readout

brain = axonweave.load("male-cns:v1.0")

model = BrainModel(
    brain,
    dynamics="rate",        # "lif", "adaptive_lif", "rate", or a custom model
    trainable_edges=False,  # Mode 1: frozen connectome
)
model.connect(Input(784))
model.connect(Readout(10))

model.fit(train_loader, epochs=3)
```

Conceptually:

```text
784 inputs → encoder → connectome dynamics → 10-neuron readout → classification
```

## ConnectomeBlock: composable nn.Module

AxonWeave does not dictate your architecture. `ConnectomeBlock` is an ordinary `torch.nn.Module` you can place between arbitrary layers:

```python
import torch.nn as nn
from axonweave.frameworks.torch import ConnectomeBlock

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    ConnectomeBlock(brain, dynamics="lif", select=256),
    nn.LayerNorm(brain.n_neurons),
    nn.Linear(brain.n_neurons, 10),
)
```

## Brain task and agent facades

The loaded brain object exposes higher-level entry points:

```python
# Supervised task
model = brain.task(
    input=axonweave.encoders.TokenEncoder(vocab_size=50_000),
    output=axonweave.decoders.TokenDecoder(vocab_size=50_000),
    dynamics="lif",
)
model.fit(dataset)

# Environment agent
agent = brain.agent(
    input=axonweave.encoders.ImageEncoder(shape=(84, 84, 3)),
    output=axonweave.decoders.ActionDecoder(actions=6),
    dynamics="lif",
    learning="stdp",
)
agent.run(environment, episodes=100)
```

:::DOC-NOTE
The connectome provides structural connectivity and published metadata. Dynamics, encoders, decoders and learning rules are explicit model choices — AxonWeave never presents them as biological facts.
:::

## Five training modes

| Mode | What trains | API |
|---|---|---|
| 1. Frozen substrate | encoders/decoders only | `BrainModel(brain, trainable_edges=False)` |
| 2. Synaptic weights | `W = W0 + ΔW`, topology preserved | `trainable_edges=True` |
| 3. Neuron parameters | τ, thresholds, leak, gain | `train_dynamics=True` |
| 4. Local plasticity | STDP / dopamine-modulated updates | `learning="stdp"` / `"dopamine_stdp"` |
| 5. Hybrid | backprop outside + plasticity inside | combine the above |

## Scientific honesty

Do not describe a trained readout as the fly brain "understanding" the task. A next-token experiment means: encoding → connectome dynamics → readout optimized against cross-entropy. AxonWeave records which components come from source data and which are model assumptions in experiment metadata.
