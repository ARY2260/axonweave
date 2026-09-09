# Checkpoints

One-sentence purpose: explain how experiment checkpoints preserve substrate identity alongside learned parameters — and how compatibility is enforced.

:::DOC-WARN
Checkpointing currently covers the agent experiment API. Substrate-fingerprint validation on load is implemented at the registry level (manifest identity) and will extend to graph-fingerprint comparison in checkpoints.
:::

## Why identity matters

A trained model is only meaningful against the substrate it was trained on. A checkpoint therefore records context, not just weights:

- AxonWeave version
- substrate ID (e.g. `male-cns:v1.0`)
- dynamics configuration (model name and parameters)
- learning rule configuration
- encoder/decoder class names
- creation timestamp

## Saving

```python
agent = brain.agent(input=..., output=..., dynamics="lif", learning="stdp")
agent.run(environment, episodes=100)
agent.save_checkpoint("runs/exp1/ckpt-0100.awb-ckpt")
```

This writes:

- `ckpt-0100.awb-ckpt.npz` — compressed weights (working copy if plasticity ran, otherwise the substrate graph)
- `ckpt-0100.awb-ckpt.json` — metadata manifest

## Loading

```python
from axonweave.experiment import Agent

restored = Agent.load_checkpoint("runs/exp1/ckpt-0100.awb-ckpt", brain)
```

The loader reconstructs the dynamics model and learning rule from the metadata and restores the weight matrix.

## The graph fingerprint concept

Every substrate build computes a SHA-256 fingerprint over the CSR data, indices, indptr and body IDs. Two substrates with different fingerprints are materially different graphs, even if their IDs match. Loading a checkpoint against an incompatible substrate must fail safely — this check is on the roadmap (see [IMPLEMENTATION_GAP](https://github.com/dhakalnirajan/axonweave/blob/main/docs/development/IMPLEMENTATION_GAP.md)).

## Reproducibility metadata

Checkpoints deliberately capture the "what was learned and why" context so experiments can be reported honestly:

- which components came from source data (the substrate)
- which were model assumptions (dynamics, plasticity configuration)
- which were learned (the weight deltas)

## Related

- [Experiments](experiment.md) — the run loop that produces checkpoints.
- [Learning & Plasticity](learning.md) — what the working copy contains.
