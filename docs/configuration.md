# Configuration

Every AxonWeave behavior that can be changed without code edits — environment variables, cache locations and install-time options — in one place.

## Cache location

Set `AXONWEAVE_HOME` to control the local substrate cache:

```bash
export AXONWEAVE_HOME=/data/axonweave
```

On Windows PowerShell:

```powershell
$env:AXONWEAVE_HOME = "D:\axonweave"
```

## Biological assumptions

Keep receptor/sign, delay, dynamics and plasticity policies in version-controlled experiment configuration. A trained checkpoint should record these policy identities.

## Layer serialization

Serialization differs per backend, following each framework's native model:

### TensorFlow/Keras

`axonweave.keras.ConnectomeLayer` implements `get_config()`, so it composes with standard Keras serialization — `model.get_config()`, `keras.models.clone_model`, and config-based rebuilds:

```python
from axonweave.keras import ConnectomeLayer

layer = ConnectomeLayer(brain.graph, trainable_edges=True, learnable_gain=True, use_bias=True)
cfg = layer.get_config()
# {'trainable_edges': True, 'learnable_gain': True, 'use_bias': True,
#  'n_neurons': <substrate size>, ...}
```

The config records:

- `trainable_edges`, `learnable_gain`, `use_bias` — the structural options you passed;
- `n_neurons` — provenance for the expected substrate dimension.

The sparse topology itself is **not** embedded in the config. A deserialized or cloned layer re-resolves the graph from the substrate, so the substrate must be installed (and loadable) when the model is rebuilt. This is deliberate: embedding edge data in a JSON config would break the wheel/substrate separation and bypass fingerprint validation. Identity is enforced where it belongs — the substrate fingerprint recorded in checkpoints and validated at load time (see [Checkpoints](checkpoints.md)).

```python
# Round trip: config -> new layer -> rebuild against the same substrate
layer2 = ConnectomeLayer(brain.graph, **{
    k: v for k, v in cfg.items() if k in
    ("trainable_edges", "learnable_gain", "use_bias")
})
assert layer2.get_config()["n_neurons"] == brain.n_neurons
```

### PyTorch

PyTorch serialization is state-dict based and needs no AxonWeave-specific config: `layer.state_dict()` captures `edge_weight`, `gain` and `bias`, and `torch.save`/`torch.load` handle persistence. The sparsity pattern (`edge_index`) is a registered buffer and travels with the state dict — guard it the same way you guard the substrate fingerprint, because a state dict is only meaningful against the topology it was trained on:

```python
torch.save(layer.state_dict(), "layer.pt")
# ...
layer.load_state_dict(torch.load("layer.pt"))
```

For full-model checkpoints (including substrate identity, backend and policy metadata), prefer the experiment checkpoint format in [Checkpoints](checkpoints.md), which validates the substrate fingerprint on load and refuses a mismatched graph with `AXW002`.

### JAX

The JAX adapter is functional: layers hold immutable arrays (`edge_weight`, `indices`, `bias`) rather than mutable parameter trees, so serialization is plain array persistence — `np.savez` / `orbax` / `flax.serialization` over the attributes you care about. There is no `get_config()` equivalent; reconstruct the layer from the same constructor arguments plus the substrate, then restore the arrays.

## What never serializes

Regardless of backend, these never travel inside a layer config or state file:

- raw upstream MaleCNS data (substrates are provisioned, not embedded);
- the CSR topology itself (always re-derived from the installed substrate);
- anything that would bypass substrate fingerprint validation.
