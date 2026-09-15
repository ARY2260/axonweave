# API Reference

Complete function-level reference for the public AxonWeave surface — loading, selection, layers, tasks and errors — with each entry mirroring the live docstrings.

## `axonweave.load`

```python
brain = axonweave.load("male-cns:v1.0")
```

Loads an installed substrate. It does not silently download multi-gigabyte data. Provisioning is explicit through the CLI.

## `ConnectomeGraph`

```python
graph.n_neurons
graph.n_edges
graph.index(body_id)
graph.neighbors(body_id)
```

The graph preserves original body IDs and stores directed connectivity as a SciPy CSR matrix.

## `BiologicalBrain`

```python
brain.graph
brain.n_neurons
brain.torch_layer(...)
brain.keras_layer(...)
brain.memory_estimate(dtype="float32", state="full")
```

The object binds the graph to optional source metadata paths and framework adapters. `memory_estimate()` reports the per-component simulation footprint (neuron state, edge parameters, delay/receptor/plasticity reservations, total) in bytes.

## `axonweave.runtime.ConnectomeRuntime`

Stateful temporal execution over the connectome (NumPy reference path). See [Temporal Runtime](runtime.md) for semantics and examples.

```python
from axonweave.runtime import ConnectomeRuntime
from axonweave.dynamics import LIF

rt = ConnectomeRuntime(brain.graph, dynamics=LIF())
rt.reset_state()                 # clear carried state (t -> 0)
y = rt.step(x_t)                 # one timestep; state persists
seq = rt.forward_sequence(x)     # [..., T, F] -> [..., N]; == sequential steps
state = rt.get_state()           # deep-copied RuntimeState snapshot
rt.set_state(state)              # restore (replay / branch)
rt.detach_state()                # BPTT boundary (no-op on NumPy path)
rt.timestep                      # scalar clock
rt.memory_estimate()             # per-component state bytes
```

State snapshots compose `NeuronState` (membrane/adaptive/refractory variables + clock), `SynapticState` (delayed signals; reserved), `PlasticityState` (traces; reserved) and `RuntimeState.timestep`. Snapshots are deep-copied on both capture and restore; `RuntimeState.to_dict()` gives a plain-structure view. `forward_sequence` is exactly equivalent to sequential `step` calls from the same initial state — enforced by tests across Rate/LIF/AdaptiveLIF.

## `axonweave.torch.BrainModel`

Declarative model composition: encoder -> stateful runtime -> readout. See [Temporal Runtime](runtime.md).

```python
from axonweave.torch import BrainModel
from axonweave.encoders import VectorEncoder
from axonweave.dynamics import SurrogateLIF
from axonweave.readout import RegressionReadout

model = BrainModel(
    brain=brain,
    encoder=VectorEncoder(input_dim=8, output_dim=256),
    dynamics=SurrogateLIF(),            # differentiable path (BPTT)
    readout=RegressionReadout(n_source=256, n_outputs=1),
)
model.reset_state()
y = model.step(x_t)                     # encoder -> runtime -> readout
seq = model.forward_sequence(x)         # [..., T, F] -> [..., T, out]
model.get_state() / model.set_state(s) / model.detach_state()
```

Exposes `model.encoder`, `model.brain`, `model.dynamics`, `model.readout` and `model.runtime`. When dynamics is `SurrogateLIF`/`SurrogateAdaptiveLIF`, the runtime is the differentiable torch cell (`TorchSurrogateLIF`): `loss.backward()` reaches edge weights, gain and the readout through the sequence, and `detach_state()` truncates BPTT. The legacy `connect(Input(...))`/`connect(Readout(...))` wiring remains supported; constructor injection takes precedence.

## `SignalPolicy`

```python
SignalPolicy(mapping={"label": 1.0}, default_gain=0.0)
```

This is an explicit model assumption. It is not an upstream biological truth.

## Error codes

`AXW001` substrate missing; `AXW002` integrity/provenance mismatch (incl. checkpoint and manifest fingerprint checks); `AXW003` schema error; `AXW004` unsupported device; `AXW005` biological assumption error; `AXW006` optional backend unavailable; `AXW007` non-fatal configuration warning; `AXW010` API/shape/selection misuse. Full table in [Errors & Diagnostics](errors.md).
