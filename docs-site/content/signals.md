# Signals & Receptors

:::DOC-WARN
Receptor models and synaptic delays are planned interfaces, not yet implemented. The `SignalPolicy` mechanism exists today.
:::

## The problem

The upstream MaleCNS release provides **aggregate neurotransmitter predictions** per neuron: e.g. "this neuron releases acetylcholine". That is a source fact.

What the dataset does **not** specify is how that transmitter changes the target neuron's state — which depends on the target's receptor composition, which varies per synapse and is not fully mapped. Converting "releases GABA" into "inhibitory" is therefore a **modeling assumption**, not data.

## AxonWeave's rule

Neurotransmitter identity never automatically becomes a universal excitatory/inhibitory rule. Effective influence requires an explicit, user-configurable policy.

## SignalPolicy

```python
from axonweave.signals import SignalPolicy

policy = SignalPolicy(
    mapping={
        "acetylcholine": 1.0,
        "gaba": -1.0,
    },
    default_gain=0.0,   # unknown transmitters contribute nothing unless you decide
)

gain = policy.gain("gaba")   # -1.0
```

This is a computational policy. It is not a statement about Drosophila biology. Record it in your experiment configuration and checkpoints.

## Planned: receptor models

The planned `ReceptorModel` interface separates transmitter identity from receptor response:

```text
neurotransmitter identity
        +
receptor type (user/model policy)
        ↓
effective signal
```

A receptor model will be replaceable; users will supply custom implementations. Until it exists, `SignalPolicy` covers sign/strength decisions.

## Planned: synaptic delays

Synaptic transmission is not instantaneous. The planned delay interface will support constant, per-edge, per-neuron-type and user-callback delay models, compatible with time-stepped dynamics.

## Related

- [Scientific Reference](scientific-reference.md) — what is source data vs assumption.
- [Neuron Dynamics](dynamics.md) — how effective signals drive state.
- [Learning & Plasticity](learning.md) — how plasticity interacts with sign.
