# Temporal Runtime

The connectome is a recurrent, stateful computational substrate. The runtime package turns the existing primitives — sparse topology, dynamics, delays, plasticity — into one temporal execution model with explicit state and deterministic replay.

## The core loop

```python
from axonweave.runtime import ConnectomeRuntime
from axonweave.dynamics import LIF

runtime = ConnectomeRuntime(brain.graph, dynamics=LIF())
runtime.reset_state()
for x_t in stream:          # x_t: [features] or [batch, features]
    y_t = runtime.step(x_t) # state persists between calls
```

`step()` advances exactly one timestep and never resets state automatically. Episodes and sessions are bounded explicitly with `reset_state()`.

## Sequence semantics

```python
seq_out = runtime.forward_sequence(x)   # x: [..., T, F] -> [..., T, N]
```

`forward_sequence` is exactly equivalent to calling `step` on each timestep from the same initial state under deterministic execution. This equivalence is enforced by tests across the `Rate`, `LIF` and `AdaptiveLIF` models — it is a contract, not an approximation.

## Explicit state

State is a first-class object, not hidden module globals:

```python
state = runtime.get_state()   # deep copy: safe to keep, branch, serialize
y = runtime.step(x)
runtime.set_state(state)      # rewind
y2 = runtime.step(x)          # identical to the first y — replay works
```

A `RuntimeState` snapshot contains:

| Component | Contents |
|---|---|
| `NeuronState` | membrane/adaptive/refractory variables, the timestep clock |
| `SynapticState` | delayed signals, receptor state (reserved; delay engine composes here) |
| `PlasticityState` | eligibility, STDP and reward traces |
| `RuntimeState.timestep` | scalar clock for the whole runtime |

Snapshots are deep-copied on both `get_state()` and `set_state()`, so captured states are never mutated by later steps, and `to_dict()` provides a plain-Python serializable view.

## Truncated BPTT

`detach_state()` marks a gradient boundary. On the NumPy reference path it is an explicit no-op (there is no autograd graph); framework adapters override it to break carried gradients across sequence chunks:

```python
for chunk in chunks:
    y = model.forward_sequence(chunk)
    loss = criterion(y, targets)
    loss.backward()
    model.detach_state()
```

## Gradients through spiking dynamics (BPTT)

When the model's dynamics is a surrogate-spiking model (`SurrogateLIF` / `SurrogateAdaptiveLIF`), the torch `BrainModel` routes the sequence through a differentiable, torch-native spiking cell instead of the reference bridge. The membrane state is rebuilt as a function of the previous state each step, so autograd builds a true recurrent graph (BPTT), and gradients reach the connectome's edge weights:

```python
from axonweave.dynamics import SurrogateLIF, SigmoidSurrogate

model = BrainModel(
    brain=brain,
    encoder=TimeSeriesEncoder(input_dim=8, output_dim=256),
    dynamics=SurrogateLIF(surrogate=SigmoidSurrogate(k=5.0)),
    readout=RegressionReadout(n_source=256, n_outputs=1),
)

out = model.forward_sequence(seq)     # [..., T, F] -> [..., T, out]
loss = criterion(out, targets)
loss.backward()                       # reaches edge weights, gain, readout
opt.step()
model.detach_state()                  # truncated BPTT boundary
```

The spike is a hard threshold forward and a surrogate derivative backward; the four surrogate families (sigmoid, atan, piecewise, straight-through) mirror the native `surrogate_backward` formulas exactly and are equivalence-tested against them. The surrogate choice is explicit — the library never silently substitutes surrogate gradients.

Sparse propagation on this path uses a persistent torch sparse matrix built once at model construction (no per-step COO rebuilds).

## High-level model

The torch `BrainModel` composes the runtime with interfaces. Construction is declarative — you pass components, the model wires them:

```python
import torch
from axonweave.torch import BrainModel
from axonweave.encoders import VectorEncoder
from axonweave.dynamics import LIF
from axonweave.readout import RegressionReadout

model = BrainModel(
    brain=brain,
    encoder=VectorEncoder(input_dim=8, output_dim=256, seed=0),
    dynamics=LIF(),
    readout=RegressionReadout(n_source=256, n_outputs=1),
)

model.reset_state()
for x_t in stream:
    prediction = model.step(x_t)          # encoder -> runtime -> readout
```

The model exposes its components (`model.encoder`, `model.brain`, `model.dynamics`, `model.readout`, `model.runtime`), supports `model.forward_sequence(sequence)`, and the same state API (`get_state`/`set_state`/`detach_state`). The legacy `connect(Input(...))`/`connect(Readout(...))` wiring keeps working; constructor injection takes precedence when both are used.

## Time-series encoding

`TimeSeriesEncoder` maps `[B, T, features]` streams per timestep. With `window=1` the mapping is memoryless — recurrence lives in the connectome, where it belongs. With `window=k` the current at step `t` carries the last `k` observations (delay-line style, zero-padded at the sequence start):

```python
from axonweave.encoders import TimeSeriesEncoder

encoder = TimeSeriesEncoder(input_dim=10, output_dim=256, window=1)
model = BrainModel(brain=brain, encoder=encoder, dynamics=LIF(), readout=...)
forecast = model.forward_sequence(history)
```

Every encoder declares `input_shape`, `output_size` and `dtype` so wiring errors surface before execution (`AXW010` otherwise).

## Memory estimation

MaleCNS-scale state is large; know the footprint before you run:

```python
brain.memory_estimate(dtype="float32", state="full")
# {'n_neurons': ..., 'n_edges': ..., 'neuron_state': ..., 'edge_parameters': ...,
#  'total': ..., ...}

runtime.memory_estimate()   # same accounting from the runtime side
```

The estimate covers neuron state, edge parameters and (once composed) delay, receptor and plasticity state. It deliberately does not encourage dense `features × 166700` projection matrices — use [Selection & Sub-Networks](examples-selection.md) and small encoder outputs to keep task interfaces modest.

## Backend status

The NumPy reference runtime is the semantic source of truth and is fully tested locally. The torch bridge (`TorchStatefulRuntime`) implements the same semantics on tensors and is CI-verified-pending, as is the differentiable spiking path (`TorchSurrogateLIF`, BPTT). See [STATUS](https://github.com/dhakalnirajan/axonweave/blob/main/docs/development/STATUS.md) for the three-state audit.

## End-to-end example

[examples/time_series_forecasting.py](https://github.com/dhakalnirajan/axonweave/blob/main/examples/time_series_forecasting.py) trains the full pipeline (TimeSeriesEncoder -> recurrent connectome -> RegressionReadout) against MLP/RNN/LSTM/GRU baselines on a synthetic multivariate task, with every component labeled as biological, assumed, learned or task-specific.
