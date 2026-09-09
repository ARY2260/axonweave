# Architecture

AxonWeave separates source data, biological assumptions and framework execution.

## Runtime flow

```text
MaleCNS source
    ↓
substrate manifest
    ↓
download + validate
    ↓
sparse substrate cache
    ↓
BiologicalBrain
    ↓
framework adapter
    ↓
native sparse/device operations
    ↓
user model
```

## Composability

AxonWeave layers are ordinary framework layers. A user can place them between arbitrary native or custom layers.

```python
model = nn.Sequential(
    custom_input_encoder,
    nn.MultiheadAttention(...),
    axonweave_layer,
    custom_dynamics_layer,
    nn.LayerNorm(...),
    custom_readout,
)
```

The exact tensor shapes are application-defined; the AxonWeave layer requires its final feature dimension to match the substrate state dimension.

## Rust boundary

Rust/PyO3 is for native kernels, graph processing and future high-performance operations. Python remains the public API. Framework tensor/device execution remains in the framework adapter whenever possible.

## Provenance

Every future trainable checkpoint should bind to a substrate fingerprint, biological-policy configuration and backend version.
