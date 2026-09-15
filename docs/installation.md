# Installation


## The three layers

1. **Python package** — the runtime (`axonweave`). Small; installs in seconds.
2. **Optional framework extras** — PyTorch, TensorFlow/Keras, JAX, neuPrint, navis. Install only what you use.
3. **Biological substrate** — the MaleCNS v1.0 connectome artifact. Multi-gigabyte; installed explicitly via the CLI. Never bundled with the wheel.

## Standard install

```bash
pip install axonweave
```

This includes NumPy, SciPy, PyArrow, requests and pydantic — enough to load substrates and run the NumPy reference path.

## Compiled compute core

Wheels for Linux, macOS and Windows ship the compiled Rust core (`axonweave._native`) built from `rust/` — installing a wheel never requires a Rust toolchain. Source and editable installs without a built extension run an identical NumPy/SciPy reference instead. Either way the public API behaves the same; check which path is active with:

```python
import axonweave
print(axonweave.load("male-cns:v1.0").info().native_backend)  # "0.2.0" (extension) or "python-fallback"
```

See [Rust Core](rust-core.md) for the dispatch contract and [Backends](backends.md) for backend coverage.

## Optional framework extras

```bash
pip install "axonweave[torch]"          # PyTorch >= 2.2
pip install "axonweave[tensorflow]"     # TensorFlow >= 2.15
pip install "axonweave[jax]"            # JAX >= 0.4.30
pip install "axonweave[neuprint]"       # neuprint-python
pip install "axonweave[navis]"          # navis
pip install "axonweave[all]"            # everything above + rich
```

`import axonweave` works even when none of these are installed. Missing backends raise `AXW006` (`BackendUnavailableError`) with an actionable message when you try to use them.

## Development install

```bash
git clone https://github.com/dhakalnirajan/axonweave.git
cd axonweave
python -m pip install -e '.[dev]'
pytest -q
```

## Substrate installation

```bash
axonweave substrate install male-cns:v1.0
```

This downloads the official MaleCNS v1.0 files, verifies them against published checksums, builds the sparse graph, and caches the result. See [Substrate Distribution](distribution.md) for cache location and options.

## Requirements

- Python 3.10–3.14
- Linux, macOS, or Windows
- Sufficient disk for the substrate (the full release includes multi-gigabyte synapse resources; the base connectivity/annotation set is smaller)

## What is NOT installed automatically

- The MaleCNS biological data. Installing the package never triggers a multi-gigabyte download.
- Any ML framework. Torch/TensorFlow/JAX are opt-in extras.

## Next steps

- [Getting Started](getting-started.md) — from zero to first forward pass.
- [Substrate Distribution](distribution.md) — cache, provenance, offline use.
