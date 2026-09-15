# Training


:::DOC-NOTE
All training modes preserve substrate identity and graph topology. Structural plasticity (adding/removing edges) is not supported.
:::

## Mode 1 — Frozen connectome

The substrate is completely fixed. Only encoders, decoders and readouts train through normal backpropagation.

```python
model = BrainModel(brain, dynamics="rate", trainable_edges=False)
```

- What trains: interface parameters.
- What stays fixed: every synaptic weight, all dynamics.
- Use when: benchmarking the substrate as a fixed feature map; cheapest and most reproducible.

## Mode 2 — Trainable synapses

Synaptic weights become trainable, initialized from biological values:

```text
W = W₀ + ΔW
```

```python
model = BrainModel(brain, dynamics="rate", trainable_edges=True)
```

- What trains: one parameter per **existing** edge, initialized at the biological weight.
- What stays fixed: topology. Non-existing edges remain exactly zero — AxonWeave never invents synapses.
- Use when: adapting connectivity signal flow to a task.

## Mode 3 — Trainable neuron parameters

Connectivity untouched; per-neuron parameters train instead:

- membrane time constant τ
- firing threshold
- leak / resting potential
- global gain and bias

```python
model = BrainModel(brain, train_dynamics=True)
```

- What trains: far fewer parameters than Mode 2; can be dramatically cheaper.
- Status: parameter plumbing implemented for gain/bias; full per-neuron τ/threshold training is planned.

## Mode 4 — Local plasticity

No backpropagation through the brain. Synapses update from local activity traces and reward:

```text
pre-activity × post-activity × neuromodulatory signal → synaptic update
```

```python
from axonweave.learning import STDP, DopamineSTDP

agent = brain.agent(..., learning="stdp")
agent = brain.agent(..., learning="dopamine_stdp")
```

- What changes: the working copy of the weights, every step inside the environment loop.
- What stays fixed: the cached substrate graph (copy-on-write; verified by tests).
- Use when: online/agent experiments where backprop is impractical.

## Mode 5 — Hybrid

Combine global gradient descent outside the substrate with local plasticity inside it:

```python
model = BrainModel(brain, trainable_edges=True, learning="dopamine_stdp")
```

Backprop trains interfaces and synaptic residuals; STDP adapts synapses locally during environment interaction.

## Comparison

| Mode | Trains | Optimizer | Substrate graph |
|---|---|---|---|
| 1 Frozen | interfaces | host framework (Adam/SGD) | untouched |
| 2 Synapses | existing edges | host framework | topology preserved |
| 3 Neuron params | τ/threshold/gain | host framework | untouched |
| 4 Plasticity | synapses locally | local rule (no backprop) | working copy |
| 5 Hybrid | interfaces + edges + synapses | both | working copy for plasticity |

## Scientific caution

These are computational training strategies applied to a biological structure. They are not biologically equivalent learning mechanisms. Plasticity rules in particular are *inspired by* biology, not established complete biological mechanisms. See [Scientific Reference](scientific-reference.md).

## Related

- [Learning & Plasticity](learning.md)
- [Framework API](framework.md)
- [Experiments](experiment.md)
