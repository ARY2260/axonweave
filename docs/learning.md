# Learning & Plasticity

AxonWeave supports five distinct training/adaptation paradigms. They are composable — hybrid training combines global gradient descent outside the substrate with local plasticity inside it.

## Mode 1 — Frozen connectome

The substrate is fixed; only encoders and decoders train via backpropagation. The cheapest and most reproducible starting point.

```python
model = BrainModel(brain, trainable_edges=False)
```

## Mode 2 — Trainable synaptic weights

Start from biological synaptic counts and learn residual weights while preserving topology:

$$W = W_0 + \Delta W$$

Existing edges become trainable; non-existing edges stay exactly zero unless you explicitly enable structural plasticity.

```python
model = BrainModel(brain, trainable_edges=True)
```

## Mode 3 — Trainable neuron parameters

Connectivity stays fixed; per-neuron or per-type parameters train instead:

- membrane time constant τ
- firing threshold
- leak / resting potential
- global gain and bias

```python
model = BrainModel(brain, train_dynamics=True)
```

## Mode 4 — Local plasticity

Updates driven by local activity and reward signals rather than backpropagation:

```python
from axonweave.learning import STDP, DopamineSTDP

rule = STDP(a_plus=0.01, a_minus=0.012, tau_pre=20.0, tau_post=20.0)
rule = DopamineSTDP(base=STDP(), dopamine_gain=1.0)   # three-factor rule
```

Three-factor updates use:

```text
pre-activity × post-activity × neuromodulatory (reward/dopamine) signal
```

## Mode 5 — Hybrid training

Global backpropagation through framework layers + local plasticity inside the connectome block:

```python
model = BrainModel(
    brain,
    trainable_edges=True,
    learning="dopamine_stdp",
)
model.fit(env_iterator)   # backprop trains interfaces; STDP adapts synapses
```

:::DOC-WARN
Plasticity updates are model assumptions, not biological facts. Upstream neurotransmitter predictions inform — but do not determine — sign and strength of plasticity. Record the chosen rule and parameters in experiment metadata.
:::

## Choosing a mode

| Goal | Recommended mode |
|---|---|
| Benchmark the substrate as a fixed feature map | 1 |
| Adapt connectivity to a task | 2 |
| Cheap personalization with few parameters | 3 |
| Online/agent experiments with reward | 4 |
| Research on combined adaptation | 5 |
