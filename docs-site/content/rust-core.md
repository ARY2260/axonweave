# Rust Core

One-sentence purpose: explain how AxonWeave's low-level compute is implemented once in Rust and dispatched through a single layer, with an identical Python fallback.

## Why a Rust core

The Rust core (`axonweave._native`, built from `rust/`) gives every computationally meaningful primitive a single, tested implementation with accelerator-grade performance. It is a **compute substrate** — never a brain simulation and never a second runtime:

```text
user model
    ↓
framework adapter (tensor/device semantics owned by PyTorch / TensorFlow / NumPy)
    ↓
native.py — single dispatch layer
    ├── _HAS_NATIVE=True  → axonweave._native (compiled PyO3 extension)
    └── _HAS_NATIVE=False → native._numpy_* (SciPy/NumPy reference)
    ↓
sparse substrate cache (CSR graph, body IDs)
```

Public layers never call the extension directly. They call functions in `native.py`, which coerce inputs to canonical dtypes (int64 indices/indptr, float32 data), run the compiled kernel or its reference twin, and reshape flat outputs back to public shapes. `native.py` never returns framework tensors.

## What ships where

| Module | Kernels |
|---|---|
| `graph.rs` | `sparse_matmul`, `sparse_matmul_transpose`, `csr_matmul_2d`, `csr_matmul_2d_transpose`, `build_csr`, `csr_submatrix`, `csr_fingerprint` |
| `dynamics.rs` | `lif_step`, `adaptive_lif_step`, `rate_step` |
| `surrogate.rs` | `surrogate_forward`, `surrogate_backward`, `surrogate_lif_step`, `surrogate_adaptive_lif_step` |
| `learning.rs` | `stdp_update` (per-edge LTP/LTD, optional reward multiplier, `w_min`/`w_max` clipping) |
| `signals.rs` | `vesicle_release_step`, `nt_currents` |
| `receptors.rs` | `receptor_step` (AMPA / GABA / NMDA / dopamine) |
| `delays.rs` | `DelayRing`, `delay_ticks_from_ms`, `apply_delays` |
| `encoders.rs` | `dense_matmul`, `row_absmax_normalize`, `embed_lookup` |
| `decoders.rs` | `decode_argmax`, `decode_clip` |
| `readout.rs` | `readout_logits` |
| `provisioning.rs` | `sha256_file`, `md5_base64_file` |

Every kernel has a matching `native._numpy_*` reference that implements the identical semantics.

## The fallback guarantee

```python
try:
    from . import _native as _NATIVE
    _HAS_NATIVE = True
except ImportError:
    _NATIVE = None
    _HAS_NATIVE = False
```

The extension is built only in CI, so source installs run the pure-Python path — and public results are the same either way. `_numpy_*` is the single numeric source of truth; the Rust kernels stay in sync with it, and `tests/test_native_runtime.py` proves the two paths are equivalent against the built wheel in CI.

:::DOC-NOTE
Local checkouts without a Rust toolchain never compile the extension. The `native-equivalence` CI job is authoritative for the compiled path.
:::

## Surrogate gradient kinds

`surrogate_backward` and siblings take `kind` with `x = v − threshold`:

| kind | name | gradient |
|---|---|---|
| `0` | sigmoid | `k·σ(xk)·(1−σ(xk))` on `clip(xk, −20, 20)` |
| `1` | atan | `k / (1 + (πk·x)²)` |
| `2` | piecewise | `k` if `|x| ≤ 1/k`, else `0` |
| `3` | STE | `1` if `|x| ≤ width`, else `0` |

Forward is always the hard threshold `v ≥ threshold`. Receptor kinds: `0` AMPA and `1` GABA are `g·V_rev`; `2` NMDA adds the voltage-dependent Mg block; `3` dopamine is modulatory (`g·gain·sign`).

## Adding a primitive

1. Rust kernel + registration in the module's `register` (listed in `lib.rs`).
2. `_numpy_*` reference in `native.py` — the numeric source of truth.
3. Public dispatcher following the dtype/1-D/reshape contract.
4. Equivalence test in `tests/test_native_runtime.py`.

## Related

- [Architecture](architecture.md) — system layout and the Rust boundary.
- [Backends](backends.md) — how `native.py` sits under the NumPy and framework adapters.
- [Release Process](release-engineering.md) — CI wheel building with the extension included.