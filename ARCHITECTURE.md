# AxonWeave Architecture

## System model

```text
                         ┌──────────────────────────┐
                         │     User Application      │
                         │ PyTorch / Keras / NumPy │
                         └────────────┬─────────────┘
                                      │ native tensors
                         ┌────────────▼─────────────┐
                         │   AxonWeave Layer API     │
                         │ graph + dynamics policy  │
                         └────────────┬─────────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
          ┌──────▼─────┐      ┌───────▼──────┐     ┌──────▼──────┐
          │ PyTorch    │      │ TensorFlow   │     │ NumPy/SciPy │
          │ native ops │      │ native ops   │     │ reference   │
          └──────┬─────┘      └───────┬──────┘     └──────┬──────┘
                 │                    │                    │
                 └────────────────────┼────────────────────┘
                                      │
                              ┌───────▼──────┐
                              │ Rust / PyO3   │
                              │ native kernels│
                              └──────────────┘

     ┌───────────────────────────────────────────────────────────┐
     │ Substrate Registry / Provisioner                         │
     │ versioned manifest → download → validate → build → cache │
     └───────────────────────────────────────────────────────────┘
                                      │
                              MaleCNS v1.0
```

## Layering

The architecture has five layers:

1. **Scientific source layer**: upstream connectome, annotations, neurotransmitter predictions, synapse/morphology resources.
2. **Substrate layer**: immutable graph identity, body-ID mapping, sparse topology and provenance.
3. **Biological-model layer**: receptor/sign policy, delays, neuron dynamics and plasticity. These are explicit model choices.
4. **Framework adapter layer**: native PyTorch, TensorFlow/Keras and NumPy/SciPy execution.
5. **Application layer**: arbitrary user models and custom layers.

## Rust core and the dispatch contract

All low-level compute is implemented in Rust (`axonweave._native`, a PyO3
extension built from `rust/`). The Python package exposes a single dispatch
layer — `python/axonweave/native.py` — that routes every primitive to the
compiled kernel when it is importable (`_HAS_NATIVE = True`).

- **Module map** (`rust/src/*.rs`): `graph` (CSR matmul/submatrix/fingerprint),
  `dynamics` (LIF, adaptive-LIF, rate), `surrogate` (sigmoid/atan/piecewise/STE
  gradients), `learning` (`stdp_update`), `signals` (vesicle release, NT
  currents, f64), `receptors` (AMPA/GABA/NMDA/dopamine kernels, f32),
  `delays` (ring engine + stateless `apply_delays`), `encoders` (dense matmul,
  row-absmax normalize, embedding lookup), `decoders` (argmax/clip),
  `readout` (linear logits), `provisioning` (SHA-256/MD5 file digests).
- **Wrapper contract**: kernels return flat 1-D arrays (or scalars); Python
  wrappers coerce inputs to canonical dtypes and reshape to public shapes. The
  public API is byte-identical whether or not the extension is present.
- **Fallback parity**: `native._numpy_*` are the authoritative reference
  implementations; when the extension is absent they back every dispatcher so
  the pure-Python path is fully functional. CI proves native/NumPy equivalence
  on the built wheel (`tests/test_native_runtime.py`).
- **No second tensor runtime**: Rust kernels only implement this library's
  primitives; frame device execution (CUDA/MPS/etc.) stays in PyTorch/TensorFlow.

## Device principle

Device execution belongs to the host backend. AxonWeave should not duplicate CUDA, MPS, XPU or TPU runtimes. The adapters create the backend's native sparse representation and use its supported operations. If a requested operation/device combination is unsupported, AxonWeave raises an actionable error.

## Data distribution

The package is separated from the substrate because the upstream data are multi-gigabyte. `substrate install` owns download, validation, construction and cache activation. A future `.awb` packed substrate can make offline deployment reproducible.

## Scientific state

A future full model should represent:

```text
node state
 + neuron type
 + morphology metadata
 + receptor state
 + membrane/dynamics state
 + incoming delayed signals

edge state
 + structural weight
 + neurotransmitter distribution
 + receptor/sign transform
 + delay
 + optional plasticity variables
```

The source graph remains identifiable even when trainable parameters evolve.

## Checkpoint identity

Every trainable checkpoint should record:

- AxonWeave version;
- substrate ID/version;
- graph fingerprint;
- model configuration;
- biological policy identifiers/versions;
- backend and framework version;
- dtype;
- training metadata.

A checkpoint must not be loaded into a materially different substrate without an explicit override.
