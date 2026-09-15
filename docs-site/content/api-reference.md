# API Reference

Complete reference for every public class and function in AxonWeave. Each entry includes what it does, what it returns, and a working code example.

## Loading substrates

### `axonweave.load substrate_id`

Loads an installed substrate and returns a `BiologicalBrain`. The substrate must already be installed via the CLI — this function never downloads data.

```python
import axonweave

brain = axonweave.load("male-cns:v1.0")
print(f"Loaded {brain.n_neurons:,} neurons")
```

Raises `AXW001` if the substrate is not installed. Raises `AXW002` if the substrate data is corrupted or was built from a different version.

### `axonweave.substrate install`

Downloads, validates and activates a substrate. Run this before `load()`.

```bash
axonweave substrate install male-cns:v1.0
axonweave substrate install male-cns:v1.0 --disk-backed  # bounded-RAM build
axonweave substrate verify male-cns:v1.0                  # check integrity
axonweave substrate status                                # list installed
```

## BiologicalBrain

The main object returned by `load()`. Binds the connectome graph to metadata and framework adapters.

```python
brain = axonweave.load("male-cns:v1.0")

brain.n_neurons              # int: number of neurons (166,700 for MaleCNS)
brain.graph                  # ConnectomeGraph: the sparse connectivity
brain.fingerprint            # str: deterministic substrate fingerprint
brain.metadata               # NeuronMetadataStore: lazy-loaded annotations

brain.info()                 # BrainInfo: structured metadata
brain.info().summary()       # str: human-readable substrate summary
brain.capabilities()         # dict: available backends, dynamics, learning rules
brain.memory_estimate()      # dict: per-component byte footprint

brain.torch_layer(...)       # PyTorch ConnectomeLayer
brain.keras_layer(...)       # Keras ConnectomeLayer
brain.layer(...)             # framework-dispatched layer
brain.task(...)              # supervised task (encoder -> brain -> readout)
brain.agent(...)             # RL agent (encoder -> brain -> decoder -> env)
```

### Memory estimation

```python
est = brain.memory_estimate(dtype="float32", state="full")
# {'neuron_state': 2_667_200, 'edge_parameters': 102_400_000, 'total': ..., ...}
```

Covers neuron membrane state, edge parameters, delay buffers, receptor state and plasticity traces. Use this to plan GPU memory before building a model.

## ConnectomeGraph

The sparse connectivity object. Preserves original body IDs from the MaleCNS dataset.

```python
graph = brain.graph

graph.n_neurons              # int: neuron count
graph.n_edges                # int: directed edge count
graph.weights                # scipy.sparse.csr_matrix: connectivity matrix
graph.body_ids               # numpy array: body ID for each matrix row

graph.index(body_id)         # int: matrix row index for a body ID
graph.neighbors(body_id)     # array: outgoing neighbor body IDs
graph.neurons                # NeuronSelection: selection API

# Selection by various criteria
graph.neurons.all()                          # all neurons
graph.neurons.ids([12345, 67890])            # specific body IDs
graph.neurons.by_type("Kenyon cell")         # by cell type
graph.neurons.by_region("mushroom body")     # by brain region
graph.neurons.by_mask(mask_array)            # boolean mask
```

## NeuronSelection

Returned by `brain.graph.neurons`. Used to restrict layers to sub-networks.

```python
sel = brain.graph.neurons.by_region("mushroom body")
len(sel)                      # number of selected neurons
sel.body_ids                  # array of selected body IDs

# Use in a layer
layer = brain.torch_layer(selection=sel)
```

Raises `AXW010` if any requested body ID does not exist in the substrate.

## ConnectomeLayer (PyTorch)

A `torch.nn.Module` that computes sparse propagation through the connectome.

```python
import torch
from axonweave.torch import ConnectomeLayer

layer = ConnectomeLayer(
    brain.graph,
    trainable_edges=False,    # True: edge weights become nn.Parameter
    learnable_gain=False,     # True: global gain trains
    bias=False,               # True: per-neuron bias trains
    selection=None,            # NeuronSelection to restrict sub-network
    signal_policy=None,        # raises AXW007 warning (structural only)
    device=None,               # 'cuda', 'mps', etc.
)

x = torch.randn(4, brain.n_neurons)   # batch of 4
y = layer(x)                           # (4, n_neurons)

print(layer)  # ConnectomeLayer(n_neurons=166700, n_edges=25600000, ...)
```

Preserves input dtype (float32 and float64). Exposes `layer.graph_weights` (the backing CSR). Raises `AXW010` on dimension mismatches and invalid selections.

## ConnectomeLayer (Keras)

A `tf.keras.layers.Layer` with the same semantics as the PyTorch version.

```python
from axonweave.keras import ConnectomeLayer

layer = ConnectomeLayer(
    brain.graph,
    trainable_edges=True,
    learnable_gain=True,
    use_bias=True,
)

config = layer.get_config()   # serializable config for cloning
```

Implements `get_config()` for Keras serialization. The sparse topology is re-resolved from the substrate at deserialization time.

## ConnectomeRuntime

Stateful temporal execution over the connectome. The NumPy reference path is the semantic source of truth.

```python
from axonweave.runtime import ConnectomeRuntime
from axonweave.dynamics import LIF

rt = ConnectomeRuntime(brain.graph, dynamics=LIF())

rt.reset_state()                  # clear state, t -> 0
y = rt.step(x_t)                  # one timestep; state persists
seq = rt.forward_sequence(x)      # [..., T, F] -> [..., N]

state = rt.get_state()            # deep-copied RuntimeState snapshot
rt.set_state(state)               # restore (replay / branch)
rt.detach_state()                 # BPTT boundary

rt.timestep                       # current clock value
rt.memory_estimate()              # per-component state bytes
```

`forward_sequence` is exactly equivalent to sequential `step` calls from the same initial state. This is enforced by tests across Rate, LIF and AdaptiveLIF.

### State objects

`get_state()` returns a `RuntimeState` containing:

- `NeuronState`: membrane potential, adaptive variables, refractory state, clock
- `SynapticState`: delayed signals, receptor state
- `PlasticityState`: STDP traces, eligibility traces, reward traces
- `timestep`: scalar clock

All snapshots are deep-copied. `to_dict()` provides a plain-Python serializable view.

## BrainModel (PyTorch)

Declarative model composition: encoder -> stateful runtime -> readout.

```python
from axonweave.torch import BrainModel
from axonweave.encoders import VectorEncoder
from axonweave.dynamics import LIF
from axonweave.readout import RegressionReadout

model = BrainModel(
    brain=brain,
    encoder=VectorEncoder(input_dim=8, output_dim=256),
    dynamics=LIF(),
    readout=RegressionReadout(n_source=256, n_outputs=1),
)

model.reset_state()
for x_t in stream:
    prediction = model.step(x_t)

# Or process a full sequence at once
seq_out = model.forward_sequence(sequence)   # [..., T, F] -> [..., T, out]

# State management
state = model.get_state()
model.set_state(state)       # replay
model.detach_state()         # truncated BPTT boundary
```

Exposes `model.encoder`, `model.brain`, `model.dynamics`, `model.readout`, `model.runtime`. When dynamics is `SurrogateLIF` or `SurrogateAdaptiveLIF`, `loss.backward()` reaches edge weights through the sequence (BPTT).

## Dynamics

### LIF (Leaky Integrate-and-Fire)

```python
from axonweave.dynamics import LIF

lif = LIF(
    tau_membrane=0.015,    # membrane time constant (seconds)
    v_threshold=-0.050,    # spike threshold (volts)
    v_rest=-0.070,         # resting potential (volts)
    dt=0.001,              # timestep (seconds)
)
```

Single-compartment LIF with absolute refractory period. No gradient path through spikes — use `SurrogateLIF` for BPTT.

### SurrogateLIF

```python
from axonweave.dynamics import SurrogateLIF, SigmoidSurrogate

lif = SurrogateLIF(
    tau_membrane=0.015,
    surrogate=SigmoidSurrogate(k=5.0),
)
```

Differentiable LIF for gradient-based training. Forward: hard threshold. Backward: surrogate derivative. Four surrogate families available: `SigmoidSurrogate`, `ATanSurrogate`, `PiecewiseLinearSurrogate`, `StraightThroughEstimator`.

### AdaptiveLIF

Adds threshold adaptation to LIF. `SurrogateAdaptiveLIF` provides the differentiable version.

### Rate

Instantaneous firing-rate model. No spikes, no refractory period. Cheapest dynamics for gradient training.

## Encoders

### VectorEncoder

```python
from axonweave.encoders import VectorEncoder

enc = VectorEncoder(input_dim=8, output_dim=256, seed=0)
print(enc.input_shape)   # (8,)
print(enc.output_size)   # 256
currents = enc(torch.randn(4, 8))   # (4, 256)
```

Fixed random projection. The default for static feature data.

### TimeSeriesEncoder

```python
from axonweave.encoders import TimeSeriesEncoder

enc = TimeSeriesEncoder(input_dim=10, output_dim=256, window=1)
print(enc.input_shape)   # (1, 10)
seq_currents = enc(torch.randn(4, 50, 10))   # (4, 50, 256)
```

Per-timestep encoder for `[B, T, features]` streams. `window=1` is memoryless; `window=k` carries a k-step delay line.

### ImageEncoder, TokenEncoder, SensorEncoder

See [Encoders & Decoders](encoders.md) for visual, text and telemetry encoding.

## Readouts

### RegressionReadout

```python
from axonweave.readout import RegressionReadout

readout = RegressionReadout(n_source=256, n_outputs=1)
prediction = readout(neural_activity)   # (batch, 1)
```

### ClassificationReadout

```python
from axonweave.readout import ClassificationReadout

readout = ClassificationReadout(n_source=256, n_classes=10)
logits = readout(neural_activity)   # (batch, 10)
```

### TokenReadout

```python
from axonweave.readout import TokenReadout

readout = TokenReadout(n_source=256, vocab_size=50_000)
logits = readout(neural_activity)   # (batch, seq, vocab_size)
```

## SignalPolicy

```python
from axonweave.signals import SignalPolicy

policy = SignalPolicy(
    mapping={"acetylcholine": 1.0, "GABA": -0.5},
    default_gain=0.0,
)
```

Maps neurotransmitter labels to signal gains. This is an explicit modeling assumption, not an upstream biological truth.

## Learning rules

```python
from axonweave.learning import STDP, DopamineSTDP

stdp = STDP(tau_plus=0.020, tau_minus=0.020, a_plus=0.01, a_minus=0.012)
dap = DopamineSTDP(tau_c=1.0, tau_d=0.25, tau_elig=0.75)

agent = brain.agent(..., learning=stdp)
agent = brain.agent(..., learning=dap)
```

STDP is pairwise trace-based. DopamineSTDP adds a reward-modulated eligibility trace. Neither claims to model specific fly synapses.

## Error codes

| Code | Meaning | Fix |
|---|---|---|
| `AXW000` | Base class / generic misuse | Read the message |
| `AXW001` | Substrate not installed | `axonweave substrate install male-cns:v1.0` |
| `AXW002` | Integrity or fingerprint mismatch | Reinstall substrate; never load mismatched checkpoints |
| `AXW003` | Schema error in source data | Check file version |
| `AXW004` | Unsupported device | Verify CUDA/MPS availability; try `device=None` |
| `AXW005` | Missing dynamics override | Register override or use default |
| `AXW006` | Optional backend not installed | `pip install "axonweave[torch]"` |
| `AXW007` | Non-fatal config warning | Read the warning; usually means an option is ignored |
| `AXW010` | API misuse (dimensions, IDs, options) | Read the message; it states expected vs. received |
| `AXW101` | CLI: unknown substrate | Use `male-cns:v1.0` |

Full details in [Errors & Diagnostics](errors.md).
