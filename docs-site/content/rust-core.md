# Rust Core

AxonWeave's low-level compute is implemented once in Rust and exposed through a PyO3 extension (`axonweave._native`, built from `rust/`). Python remains the public API; the Rust core is the compute substrate underneath it.

## Purpose and principles

The Rust core exists to give every computationally meaningful primitive a single, tested implementation with optional accelerator-grade performance — not to become a second runtime, a brain simulation, or a shadow API surface.

- **Scientific honesty.** The connectome plus the Rust core is a compute substrate, never a claim of a complete biophysical brain simulation. Dynamics, signaling, plasticity and receptor models remain explicit, configurable modeling choices.
- **No silent divergence.** When the compiled extension is present it is used; when it is absent an identical NumPy/SciPy reference runs. Public results are the same either way, and equivalence is proven in CI.
- **One runtime per layer.** The Rust core owns low-level kernels. NumPy/SciPy owns the pure-Python reference path. Framework adapters own device/tensor semantics. Nothing introduces a second tensor runtime inside AxonWeave.

## System interplay

```text
user model
    ↓
framework adapter (tensor/device semantics owned by PyTorch/TensorFlow/NumPy)
    ↓
native.py  — single dispatch layer
    ├── _HAS_NATIVE:  -> axonweave._native (compiled PyO3 extension, rust/)
    └── !_HAS_NATIVE: -> native._numpy_* (SciPy/NumPy reference)
    ↓
sparse substrate cache (CSR graph, body IDs)
```

`native.py` is the only module that talks to the extension. Public layers never import `_native` directly; they call `native.<fn>`, which selects the compiled kernel or its reference twin. Framework adapters sit above this and never receive framework tensors from `native.py` — they own all conversion to/from their device and tensor types.

## Module map

Each Rust module registers a fixed set of pyfunctions (plus one pyclass in `delays.rs`). Every kernel has a matching `native._numpy_*` reference; the function names below are the registered extension names.

| Rust module | Kernels / classes | numpy reference |
|---|---|---|
| `graph.rs` | `sparse_matmul`, `sparse_matmul_transpose`, `csr_matmul_2d`, `csr_matmul_2d_transpose`, `build_csr`, `csr_submatrix`, `csr_fingerprint` | `_numpy_sparse_matmul`, `_numpy_sparse_matmul_transpose`, `_numpy_csr_matmul_2d`, `_numpy_csr_matmul_2d_transpose`, `_numpy_build_csr`, `_numpy_csr_submatrix`, `_numpy_csr_fingerprint` |
| `dynamics.rs` | `lif_step`, `adaptive_lif_step`, `rate_step` | `_numpy_lif_step`, `_numpy_adaptive_lif_step`, `_numpy_rate_step` |
| `surrogate.rs` | `surrogate_forward`, `surrogate_backward`, `surrogate_lif_step`, `surrogate_adaptive_lif_step` | `_numpy_surrogate_forward`, `_numpy_surrogate_backward`, `_numpy_surrogate_lif_step`, `_numpy_surrogate_adaptive_lif_step` |
| `learning.rs` | `stdp_update` | `_numpy_stdp_update` |
| `signals.rs` | `vesicle_release_step`, `nt_currents` | `_numpy_vesicle_release_step`, `_numpy_nt_currents` |
| `receptors.rs` | `receptor_step` | `_numpy_receptor_step` |
| `delays.rs` | `DelayRing` (pyclass), `delay_ticks_from_ms`, `apply_delays` | `NumpyDelayBuffer`, `_numpy_delay_ticks`, `_numpy_apply_delayed_propagation` |
| `encoders.rs` | `dense_matmul`, `row_absmax_normalize`, `embed_lookup` | `_numpy_dense_matmul`, `_numpy_row_absmax_normalize`, `_numpy_embed_lookup` |
| `decoders.rs` | `decode_argmax`, `decode_clip` | `_numpy_decode_argmax`, `_numpy_decode_clip` |
| `readout.rs` | `readout_logits` | `_numpy_readout_logits` |
| `provisioning.rs` | `sha256_file`, `md5_base64_file` | Python `hashlib` fallbacks in `native.py` |

Dependency notes: kernels are individually thread-released (`py.allow_threads`) where the compute is pure. `graph.rs` uses the `sha2` crate directly for `csr_fingerprint`; `provisioning.rs` uses `sha2`, `md-5` and `base64` for chunked (8 MiB) file hashing. Cargo dependencies are `pyo3 0.22`, `numpy 0.22`, `sha2 0.10`, `md-5 0.10` and `base64 0.22` (`rust/Cargo.toml`).

## The dispatch contract

`native.py` is the single dispatch layer. Every public dispatcher:

1. Coerces inputs to canonical dtypes — float32 data (`float32`), int64 indices/indptr and body IDs (`int64`), per-domain dtypes for signals (float64) — and flattens to 1-D where the kernel expects flat arrays.
2. Calls `_NATIVE.<fn>` when `_HAS_NATIVE` is true, otherwise calls `native._numpy_*`.
3. Reshapes flat kernel output back to the public shape (e.g. `(batch, n_rows)` for `csr_matmul_2d`) and returns NumPy arrays or scalars. `native.py` never returns framework tensors.

Kernel signatures that consume CSR always receive raw `data`/`indices`/`indptr` arrays plus explicit `n_rows`/`n_cols`, never a SciPy matrix object. Public `native` functions take the matrix and decompose it; the reference implementations use it directly.

## Fallback rationale

```python
try:
    from . import _native as _NATIVE
    _HAS_NATIVE = True
except ImportError:
    _NATIVE = None
    _HAS_NATIVE = False
```

The extension is built only in CI, so source installs and local checkouts without a Rust toolchain run the pure-Python path. Both paths are first-class:

- The `_numpy_*` functions implement the reference semantics unconditionally and are the single numeric source of truth — they stay in sync with the Rust kernels.
- Public behavior is identical with or without the extension; `tests/test_native_runtime.py` proves it (skipped when `_HAS_NATIVE` is false, run against the built wheel in CI).

## Surrogate gradient kind codes

`surrogate_backward` (and its siblings) accept `kind` in `0..3`; the argument is `x = v - threshold`.

| kind | name | formula |
|---|---|---|
| `0` | sigmoid | `k · σ(xk) · (1 − σ(xk))`, with `σ(·)` the logistic function on `clip(xk, −20, 20)` |
| `1` | atan | `k / (1 + (π·k·x)²)` |
| `2` | piecewise | `k` if `|x| ≤ 1/k`, else `0` |
| `3` | STE | `1` if `|x| ≤ width`, else `0` |

`surrogate_forward` is the hard threshold `v ≥ threshold ? 1 : 0`, shared by all kinds. Unknown kinds return `0` from the Rust kernel and raise `ValueError` from the reference — a known asymmetry, kept because the dispatcher validates kinds before reaching either.

## Receptor kind codes

`receptor_step` accepts `kind` in `0..3`; conductance decays as `g' = g + (pre − g) · (dt / τ)` for all kinds.

| kind | name | driving current |
|---|---|---|
| `0` | AMPA | `g · V_rev` |
| `1` | GABA | `g · V_rev` |
| `2` | NMDA | `g · (1 / (1 + [Mg²⁺]·exp(−slope·(V − offset)))) · V_rev` — voltage-dependent Mg block |
| `3` | dopamine | `g · gain · sign` — modulatory, no reversal potential |

## Delay ring semantics

`DelayRing` (Rust pyclass; `NumpyDelayBuffer` in NumPy) is a per-neuron circular buffer. Constructor takes `n_slots` (number of tick rows) and `n_neurons`. Interface:

- `reset()` — zero buffers and write positions.
- `step(pre, pre_idx, post_idx, delay_ticks, weights)` — write `pre` into the current slot, advance `write_pos`, then read.
- `read(pre_idx, post_idx, delay_ticks, weights)` — read without writing.
- `write_pos` getter — per-neuron write counters.

A read looks up each edge's source slot at `write_pos − delay_ticks − 1` (mod `n_slots`), accumulating `pre[slot][pre_idx[k]] · weight[k]` into `post_idx[k]`. `delay_ticks_from_ms` converts milliseconds to integer ticks by rounding and clamping to `[0, n_slots − 1]` where `n_slots = ceil(max_delay_ms/dt) + 1`. `apply_delays` is the stateless single-shot form used by the public propagation path.

## Graph construction performance

There are two graph-build paths with byte-identical output (same CSR, same `csr_fingerprint`):

- **In-memory** (`data/builder.py`, default): batches are concatenated in RAM before the CSR reduction.
- **Disk-backed** (`data/streaming_builder.py`, `axonweave substrate install --disk-backed`): edges stream to scratch memmaps batch-by-batch and are reduced via `native.build_csr_from_coo`, so Python memory stays O(batch) regardless of edge count.

The compiled `build_csr` kernel is the hot reduction in the disk-backed path whenever the native core is present. The benchmark suite `benchmarks/bench_graph_build.py` measures both paths (wall time, Python allocation peak, peak RSS) and hard-fails if their fingerprints ever diverge — a correctness tripwire, not just a performance report. Rows can be appended to `benchmarks/results.jsonl` with `--jsonl` for longitudinal tracking.

**Rust streaming roadmap** (PLAN.md Phase 4): native edge-LUT mapping and ID scanning, direct Arrow/IPC ingestion in Rust (arrow-rs), and parallel (rayon) CSR reduction — each kernel landing behind the standard contract (Rust kernel + `_numpy_*` reference + dispatcher + equivalence test) and gated on the benchmark's acceptance criteria: identical fingerprints, ≥2× wall-time improvement at MaleCNS scale, and peak RSS flat in edge count.

## Adding a new primitive

1. Write the Rust kernel in the appropriate `rust/src/*.rs` and register the pyfunction in that module's `register` (listed in `lib.rs`).
2. Add the `_numpy_*` reference in `native.py` — it stays the single numeric source of truth.
3. Add the public dispatcher in `native.py` following the dtype/1-D/reshape contract above.
4. Add an equivalence test in `tests/test_native_runtime.py` calling `_NATIVE.<fn>` directly against `native._numpy_*`.
5. CI (below) is the only place the compiled kernel and its reference are validated side by side; do not claim local verification of the extension.

## CI verification

`.github/workflows/rust.yml` runs a Rust job (cargo test + maturin build across OSes) and a `native-equivalence` job: it builds the wheel with maturin, installs it, runs `tests/test_native_runtime.py` against the compiled extension, then runs the full suite against the wheel. Locally on this machine there is no Rust toolchain, so the extension is never compiled here — CI is authoritative for the compiled path.

## Limitations

- The extension is optional by design and built only in CI; there is no local Rust toolchain on this machine.
- Rust kernels return flat arrays/scalars; all shape and dtype correctness responsibility sits in the Python wrappers.
- The Rust core is a compute substrate. It does not make the connectome a complete biophysical brain simulation; scientific-model claims remain governed by the [Scientific Reference](scientific-reference.md) rules.

## Related

- [Architecture](architecture.md) — system layout and the Rust boundary.
- [Backends](backends.md) — how `native.py` sits under the NumPy and framework adapters.
- [Release Engineering](release-engineering.md) — CI wheel building with the extension included.