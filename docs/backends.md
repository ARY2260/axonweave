# Backends and Devices

## Supported package integrations

### NumPy/SciPy

Reference CPU implementation. All compute first routes through `native.py`, which dispatches either to the compiled Rust/PyO3 extension or an identical NumPy/SciPy reference — public results are the same either way, and both paths are first-class. Useful for correctness tests, inspection and small graph experiments.

### PyTorch

Installed with:

```bash
pip install "axonweave[torch]"
```

The integration exposes `torch.nn.Module` and uses PyTorch sparse operations.

### TensorFlow/Keras

Installed with:

```bash
pip install "axonweave[tensorflow]"
```

The integration exposes a Keras `Layer` and uses TensorFlow sparse operations.

## Device model

AxonWeave does not maintain a second device abstraction. The host framework controls placement.

For PyTorch this can include CPU, CUDA, MPS, XPU and other devices supported by the installed PyTorch release. For TensorFlow it can include CPU, GPU and TPU configurations supported by TensorFlow and the specific sparse operator.

:::DOC-WARN
A device being recognized by a framework does not imply that every sparse operation used by AxonWeave is implemented on that device. Capability must be tested. Unsupported combinations must fail explicitly instead of silently copying data to CPU.
:::

## Backend contract

A backend adapter must provide:

- native tensor/layer type;
- sparse graph representation;
- gradient propagation where trainable;
- device placement through native framework APIs;
- dtype policy;
- actionable capability errors.

## Future/optional JAX integration

JAX support is an explicit optional target rather than a claim that every JAX sparse primitive is equivalent across accelerators. A JAX adapter should be added only with numerical tests for the supported sparse path and device set.
