# AGENTS.md — AxonWeave AI Agent Contract

This file is the primary operating contract for AI coding agents working on AxonWeave.

## Mission

Build a scientifically honest, production-quality Python library that exposes the MaleCNS v1.0 connectome as a reusable sparse biological neural substrate and composable framework-native layer.

## Non-negotiable principles

1. Never fabricate biological facts.
2. Never silently turn a predicted neurotransmitter into an excitatory/inhibitory truth.
3. Never claim the connectome alone is a complete biophysical brain simulation.
4. Preserve source provenance and substrate identity.
5. Prefer framework-native tensor/device execution over a parallel fake runtime.
6. Keep the PyPI wheel independent from multi-gigabyte biological data.
7. Do not break arbitrary native/custom layer composition.
8. Errors must be actionable and carry stable `AXW###` codes where appropriate.
9. New scientific assumptions require explicit configuration and documentation.
10. Every behavior change requires tests and documentation.

## Before coding

Read, in order:

- `PLAN.md`
- `ARCHITECTURE.md`
- `CHECKLIST.md`
- `CODE_TOKENS.md` (repository root) for Python/library work
- `docs-site/AGENTS.md` and `docs-site/DESIGN.md` for documentation frontend work

## Library agent rules

### API rules

Use typed public interfaces, stable error codes and explicit configuration. Avoid hidden global state.

### Data rules

Never add raw upstream MaleCNS data to source control. Substrate artifacts are provisioned and cached separately.

### Backend rules

Adapters must use the host framework's tensor, sparse and device APIs. Do not implement a second tensor runtime inside AxonWeave.

### Scientific rules

A source observation, a fitted parameter and a modeling assumption must have separate representations.

### Testing rules

Test graph shape/index invariants, serialization, error behavior, backend numerical behavior, and device capability reporting independently.

## Repository boundaries

- `python/axonweave/`: public Python package.
- `rust/`: native implementation exposed through PyO3.
- `tests/`: unit/integration tests.
- `docs/`: source technical documentation.
- `docs-site/`: actual web documentation application (has its own `AGENTS.md`).
- `.github/workflows/`: CI/CD only.

## Change protocol

For each change:

1. State the invariant being preserved.
2. Implement the smallest coherent change.
3. Add or update tests.
4. Run formatting/lint/type checks when available.
5. Run framework-specific tests when the backend is installed.
6. Update documentation and API examples.
7. Update `CHECKLIST.md` and `PLAN.md` if scope/status changes.

## Never do this

- Do not commit API keys, tokens, cloud credentials or private URLs.
- Do not place generated build directories in source control.
- Do not add large raw MaleCNS data to the Git repository.
- Do not introduce a generic “AI dashboard” visual design into the docs site.
- Do not use fake device support based only on a string such as `device="cuda"`.
- Do not make a breaking API change without a migration note.
