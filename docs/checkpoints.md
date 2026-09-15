# Checkpoints

:::DOC-TIP
Checkpoints record substrate identity alongside learned parameters. Graph-fingerprint validation is enforced at load time: restoring against an incompatible connectome fails with `AXW002` rather than silently producing a meaningless model. For how trained layer weights serialize within each framework, see [Configuration](configuration.md).
:::

## Why identity matters

A trained model is only meaningful against the substrate it was trained on. A checkpoint therefore records context, not just weights:

- AxonWeave version
- substrate ID (e.g. `male-cns:v1.0`)
- graph fingerprint (SHA-256 over the substrate's CSR payload and body IDs)
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
- `ckpt-0100.awb-ckpt.json` — metadata manifest, including the `graph_fingerprint` of the substrate the agent trained against

## Loading

```python
from axonweave.experiment import Agent

restored = Agent.load_checkpoint("runs/exp1/ckpt-0100.awb-ckpt", brain)
```

The loader reconstructs the dynamics model and learning rule from the metadata and restores the weight matrix.

## Fingerprint validation on load

Before any weights are restored, the loader re-derives the loaded brain's graph fingerprint and compares it to the one recorded in the checkpoint manifest. A mismatch raises:

```text
AXW002: checkpoint graph fingerprint does not match the loaded substrate;
refusing to restore against an incompatible connectome
```

Two substrates with different fingerprints are materially different graphs, even if their IDs match — so this check fails safely instead of producing a model whose weights point at the wrong neurons. The same fingerprint is also recorded in the substrate manifest at install time and re-verified by `SubstrateRegistry.load()` and `axonweave substrate verify`, so identity is enforced end to end: download checksums → build fingerprint → install manifest → checkpoint → load.

## Relationship to framework serialization

Checkpoints and framework serialization solve different halves of the problem:

- **Checkpoints** (this page) capture experiment identity: which substrate, which policies, what was learned. They are backend-neutral and validate the substrate fingerprint.
- **Framework serialization** (PyTorch `state_dict`, Keras `get_config()`, JAX array persistence — see [Configuration](configuration.md)) captures layer weights and structural options inside one framework's native format.

A reproducible result records both: the checkpoint for scientific identity, the framework artifact for exact weight restoration.

## Reproducibility metadata

Checkpoints deliberately capture the "what was learned and why" context so experiments can be reported honestly:

- which components came from source data (the substrate)
- which were model assumptions (dynamics, plasticity configuration)
- which were learned (the weight deltas)

## Related

- [Experiments](experiment.md) — the run loop that produces checkpoints.
- [Learning & Plasticity](learning.md) — what the working copy contains.
- [Configuration](configuration.md) — framework-level layer serialization.
