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
native.py dispatch layer
    ├── compiled extension (Rust/PyO3)
    └── NumPy/SciPy reference (identical results)
    ↓
framework adapter (tensor/device semantics)
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

## Rust core

All low-level compute now routes through a single dispatch layer (`native.py`) that either calls the compiled PyO3 extension (`axonweave._native`, built from `rust/`) or an identical NumPy/SciPy reference — public results are the same either way, proven by `tests/test_native_runtime.py` in CI. The Rust core is a compute substrate, not a second runtime or a brain simulation. See [Rust Core](rust-core.md).

## Provenance

Every future trainable checkpoint should bind to a substrate fingerprint, biological-policy configuration and backend version.
