# Biological Model

AxonWeave is deliberately conservative about biological interpretation.

## Structural substrate

The MaleCNS graph supplies neuron identities and directed connectivity. AxonWeave preserves body-ID identity and sparse topology.

## Neurotransmitters

The upstream release provides aggregate neurotransmitter predictions. AxonWeave stores these as source facts.

It does **not** hard-code:

```text
neurotransmitter X = always excitatory
neurotransmitter Y = always inhibitory
```

Instead, a user can provide a receptor/sign policy:

```python
policy = SignalPolicy(
    mapping={
        "acetylcholine": 1.0,
        "gaba": -1.0,
    },
    default_gain=0.0,
)
```

Those values are model assumptions and must be documented with the experiment.

## Future biological layers

The model roadmap includes:

- receptor-specific transforms;
- neuron-type dynamics;
- synaptic delays;
- synapse-level neurotransmitter state;
- plasticity;
- morphology-aware computation;
- distributed partitioning.

A scientific claim must identify whether a value comes from the upstream dataset, literature, fitted training, or a user-defined assumption.
