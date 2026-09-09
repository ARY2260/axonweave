# Contributing

Contributions should preserve scientific provenance, API stability and reproducibility.

## Development

```bash
python -m pip install -e '.[dev]'
pytest -q
```

For backend tests:

```bash
python -m pip install -e '.[torch]'
python -m pip install -e '.[tensorflow]'
```

## Pull requests

A PR should include tests for changed behavior, documentation for public API changes, and a note about scientific assumptions if biological semantics changed.

Do not commit raw MaleCNS data, credentials, build directories, local caches or generated documentation output.
