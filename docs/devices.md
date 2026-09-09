# Device Support

One-sentence purpose: state factually which devices work and why — AxonWeave delegates execution to the host framework, and accelerator compatibility depends on that framework's sparse-operator support.

## The device principle

AxonWeave does not maintain its own device runtime. Device placement, transfer and execution belong entirely to the host framework:

- **PyTorch** — `torch.device`: CPU, CUDA, MPS, XPU, as supported by the installed PyTorch release.
- **TensorFlow** — native TensorFlow placement: CPU, GPU, TPU where available.
- **JAX** — native JAX device semantics (planned adapter).
- **NumPy/SciPy** — CPU reference path only.

## Capability table

| Backend | CPU | CUDA | MPS | XPU | TPU | Sparse ops on accelerator |
|---|---|---|---|---|---|---|
| NumPy/SciPy (reference) | Yes | — | — | — | — | N/A (CPU only) |
| PyTorch | Yes | Framework-dependent | Framework-dependent | Framework-dependent | — | `torch.sparse.mm` support varies by device/dtype |
| TensorFlow/Keras | Yes | Framework-dependent | — | — | Framework-dependent | `tf.sparse.sparse_dense_matmul` support varies |
| JAX | Planned | Planned | — | — | Planned | — |

"Framework-dependent" means: the operation may work, but AxonWeave does not guarantee it. Test your specific device/operation/dtype combination.

## Explicit failure, never silent fallback

If a requested device cannot execute a required sparse operation, AxonWeave raises:

```text
AXW004: backend rejected device=...: <framework error>
```

AxonWeave does not silently copy tensors to CPU. If you want CPU execution, request it explicitly.

## Requesting devices

```python
# PyTorch
layer = brain.torch_layer(device="cuda")   # raises AXW004 if unsupported
```

```python
# TensorFlow — place via the framework
with tf.device("/GPU:0"):
    y = layer(x)
```

## Honest limitations

- A device being *recognized* by a framework does not imply every sparse operator is implemented on it.
- Sparse-gradient support on accelerators varies significantly across framework versions.
- The NumPy reference path is CPU-only by design — it exists for correctness and education.

See [Backends](backends.md) for backend comparison and [Troubleshooting](troubleshooting.md) for `AXW004` diagnostics.
