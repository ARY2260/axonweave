# Release Engineering

How AxonWeave is built, tested and published: CI matrices, wheel building with the native core, and what stays out of source control.

## CI matrix

The repository defines CI for Ubuntu, macOS and Windows and Python 3.10–3.14. Separate jobs build/test Rust/PyO3 and build platform wheels using maturin. Source distributions remain usable without a Rust toolchain — the pure-Python fallback in `native.py` provides identical functionality.

## PyPI publishing

Publishing uses GitHub Actions with PyPI Trusted Publishing/OIDC. No long-lived PyPI API token belongs in repository secrets.

Before enabling production publishing, configure the PyPI trusted publisher to match the repository, workflow and environment.

## Release flow

```text
tag vX.Y.Z
  ↓
quality gates
  ↓
platform wheel builds
  ↓
source distribution
  ↓
artifact inspection
  ↓
PyPI trusted publishing
  ↓
release notes
```
