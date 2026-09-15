# Core Concepts

The mental model behind AxonWeave: what the biological substrate provides, what the framework adds on top, and where your task plugs in.

## The composition model

```text
substrate  +  computational model  +  interface  +  task
```

| Concept | What it is | Who provides it |
|---|---|---|
| **Substrate** | The versioned connectome: sparse graph, body IDs, provenance | Upstream science (MaleCNS) + AxonWeave provisioning |
| **Computational model** | Dynamics, signaling rules, plasticity — how activity evolves | AxonWeave (configurable, replaceable) |
| **Interface** | Encoders and decoders bridging application data and neural state | AxonWeave (composable) |
| **Task** | The objective: classification, token prediction, control | You |

## Connectome

The wiring diagram: ~166,700 neurons and ~25.6 million directed connections (MaleCNS v1.0 retained graph). Stored as a sparse CSR matrix; never densified. See [Connectome](connectome.md).

## Substrate and BiologicalBrain

`axonweave.load("male-cns:v1.0")` returns a `BiologicalBrain` — a computational wrapper binding the graph to optional metadata (annotations, neurotransmitters) and framework adapters. See [Biological Brain](brain.md).

## Neural state and dynamics

A connectome alone is wiring, not computation. Dynamics define how per-neuron state evolves each time step: `Rate` (instantaneous), `LIF` (leaky integrate-and-fire with refractory periods), `AdaptiveLIF` (dynamic thresholds). Dynamics are model choices, not biological facts. See [Neuron Dynamics](dynamics.md).

## Signal policy

Upstream neurotransmitter predictions are data. How a neurotransmitter influences a target neuron depends on receptor composition — which the dataset does not fully specify. AxonWeave therefore requires an explicit `SignalPolicy` mapping neurotransmitter labels to gains. Neurotransmitter identity never automatically becomes an excitatory/inhibitory rule. See [Scientific Reference](scientific-reference.md).

## Plasticity

Local update rules (STDP, dopamine-modulated three-factor learning) that change synaptic weights from activity and reward — as an alternative or complement to backpropagation. Always explicitly attached; never active by default. See [Learning & Plasticity](learning.md).

## Encoder

Maps application data (images, token IDs, sensor vectors) into neural input currents. The biological brain does not inherently accept tensors; the encoder is the bridge. See [Encoders & Decoders](encoders.md).

## Decoder / readout

Maps neural activity back to application output: action indices, token logits, classifications. See [Encoders & Decoders](encoders.md).

## Task

A supervised objective wired as `input encoder → brain → readout` with a loss. Created via `brain.task(...)`. See [Framework API](framework.md).

## Agent and environment

For interactive experiments: `environment → observation → encoder → brain → decoder → action → environment → reward → plasticity`. Created via `brain.agent(...)`. The environment is framework-neutral (`reset()`/`step()`); Gymnasium adapters are optional. See [Experiments](experiment.md).

## What stays outside AxonWeave

- Tensor execution, autodiff, optimizers, accelerators — owned by the host framework (PyTorch/TensorFlow/JAX/NumPy).
- Heavy graph kernels — owned by the Rust/PyO3 native layer, used selectively.
- Your task definition — never dictated by the library.

## Next steps

- [Getting Started](getting-started.md)
- [Framework API](framework.md)
- [Scientific Reference](scientific-reference.md)
