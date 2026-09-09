#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q python
pytest -q
cargo check --manifest-path rust/Cargo.toml
