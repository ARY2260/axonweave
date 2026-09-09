# AxonWeave Implementation Checklist

Status legend:

- **COMPLETED** — implemented; locally verified where the environment allows.
- **WRITTEN (CI-VERIFIED PENDING)** — code and tests exist; execution requires
  GitHub Actions (torch/tensorflow backends, Rust, cross-platform matrix).
  "Written" is not "verified".
- **NOT STARTED** — see PLAN.md for ordering.

## COMPLETED

### Baseline
- [x] Complete repository README.
- [x] Architecture and AI-agent governance documents (`AGENTS.md`, `CODE_TOKENS.md`, docs-site `AGENTS.md`).
- [x] Frontend design system (`docs-site/DESIGN.md`, Google DESIGN.md spec, token linting).
- [x] Documentation typography: IBM Plex Sans (UI/body/headings) + Iosevka Charon Mono (code) loaded via Google Fonts with preconnect and `display=swap`; `--font-ui`/`--font-code` CSS variables with full system fallbacks; tabular-nums for API-reference tables; DESIGN.md + tokens.css kept in sync (lint 0 errors).
- [x] SVG logo and favicon.
- [x] Browser-based Markdown documentation renderer.
- [x] Sidebar, navigation, TOC, theme switcher and copy buttons.
- [x] Getting Started through Scientific Reference documentation.
- [x] Privacy, Terms and Cookie Consent UI.
- [x] 404 page.
- [x] robots.txt and sitemap.xml.
- [x] Open Graph metadata and favicon.
- [x] Responsive documentation layout.
- [x] PyTorch/TensorFlow/NumPy/SciPy packaging extras.
- [x] `.gitignore` covering caches, builds and secrets.

### Provisioning and data pipeline
- [x] Resumable downloads (`.part` files, HTTP Range requests).
- [x] Stable upstream checksum registry (GCS MD5 per file) enforced at install; sha256 recorded.
- [x] `AXW002` integrity abort before substrate activation on checksum mismatch.
- [x] Graph builder with schema alias resolution, fingerprints, deterministic builds.
- [x] Registry identity/status validation (`AXW001`, `AXW002`).
- [x] CLI `axonweave substrate install male-cns:v1.0`.

### Backend layers
- [x] NumPy/SciPy reference layer with ND batched input.
- [x] PyTorch sparse `ConnectomeLayer` (trainable edges, gain, bias, AXW004 device errors).
- [x] TensorFlow/Keras sparse `ConnectomeLayer`.
- [x] Cross-backend numerical equivalence tests (NumPy = PyTorch = Keras).
- [x] Ruff lint clean across `python/` and `tests/`.

### Connectome computing framework (Phases 1–5)
- [x] Dynamics: `LIF`, `AdaptiveLIF`, `Rate`, `DynamicsPolicy` (per-type overrides, AXW005 on unknown types).
- [x] Encoders: `ImageEncoder`, `TokenEncoder`, `SensorEncoder` (batched, seeded, AXW010 shape/vocab errors).
- [x] Decoders: `ActionDecoder` (discrete/continuous), `TokenDecoder`, `ClassificationHead`.
- [x] Learning: `STDP`, `DopamineSTDP` (three-factor reward modulation, weight clipping, topology preservation).
- [x] Experiment `Agent` loop: encode → dynamics → decode → action → env → reward → plasticity.
- [x] Copy-on-write plasticity (cached substrate graph never mutated).
- [x] JSON-lines experiment logging.
- [x] Agent checkpoints (weights + substrate/dynamics/learning metadata).
- [x] Brain facades: `brain.task(...)`, `brain.agent(...)`, `brain.layer(...)` alias, `brain.simulate`, `brain.experiment`.
- [x] `AXW006` actionable error when torch is missing for `brain.task`.
- [x] Implementation gap analysis (`docs/development/IMPLEMENTATION_GAP.md`).

### Documentation site (design-spec UI)
- [x] Structural tabs: Learn / API / Tutorials / GitHub.
- [x] Visible search trigger with `Shift+/` global shortcut and Esc to close.
- [x] Stable/nightly version selector menu.
- [x] Resizable left sidebar (drag handle, width persisted to localStorage).
- [x] Scroll-tracked right TOC with active-section highlighting.
- [x] Always-dark code blocks (theme isolation) with Prism syntax highlighting (Python/Bash/TOML).
- [x] Accent-bar callouts (Note/Constraint/Tip) replacing plain blockquotes.
- [x] MathJax auto-load for pages containing math.
- [x] Spec palette: #121212 dark / #f8f9fa light, #ee4c2c brand accent reserved for active states.
- [x] New doc pages: Framework API, Neuron Dynamics, Encoders & Decoders, Learning & Plasticity, Experiments.
- [x] Three-column grid regression fixed (resizer removed from grid flow).
- [x] Project URLs updated to `dhakalnirajan/axonweave` (topbar, pyproject, canonical, sitemap).

### CI/CD
- [x] Python 3.10–3.14 CI matrix definition.
- [x] Ubuntu/macOS/Windows CI matrix definition.
- [x] Backend smoke jobs (torch, tensorflow) running the full test suite.
- [x] Wheel build + wheel import smoke test in CI.
- [x] Source-distribution build check.
- [x] Rust/PyO3 CI.
- [x] Docs CI: typecheck → design-token lint → build → artifact verification.
- [x] Single consolidated GitHub Pages deploy workflow with `configure-pages(enablement: true)`.
- [x] `.gitignore` reviewed; no secrets, no raw data, no build artifacts committed.

## WRITTEN (CI-VERIFIED PENDING)

The following are implemented with tests authored but **not executed locally**
(the machine has no torch/tensorflow/Rust toolchain in PATH). GitHub Actions is
the authoritative validation environment:

- [ ] PyTorch `ConnectomeLayer` test suite (`tests/test_torch_layer.py`).
- [ ] TensorFlow/Keras `ConnectomeLayer` test suite (`tests/test_keras_layer.py`).
- [ ] Cross-backend equivalence tests requiring torch/tensorflow (`tests/test_cross_backend.py`).
- [ ] Torch high-level API: `BrainModel`, `ConnectomeBlock` fit/composition/training-mode tests (`tests/test_torch_brain_model.py`).
- [ ] `brain.task` happy path and `brain.layer` alias (torch-dependent).
- [ ] Rust/PyO3 build and tests on CI runners.
- [ ] Wheel builds across the OS × Python matrix.
- [ ] GitHub Pages deployment job (also blocked on repo visibility/Pages plan).

## NOT STARTED

- [ ] Run all CI jobs on GitHub and fix runner-specific failures.
- [ ] Verify Python 3.14 compatibility for every dependency/backend.
- [ ] GPU/TPU self-hosted or vendor runners; sparse-kernel tests on accelerators.
- [ ] Validate exact upstream schemas and release fixtures (incl. real annotation column names for `by_type`/`by_region` alias table).
- [ ] Replace in-memory graph assembly with disk-backed/streaming build.
- [ ] Receptor model interface, synaptic delay engine, synapse-level neurotransmitter model.
- [ ] Portable `.awb` substrate pack/unpack.
- [ ] API-doc generation from Python docstrings.
- [ ] Versioned documentation.
- [ ] Keras high-level adapter; JAX adapter.
- [ ] Surrogate-gradient training through spiking dynamics.
- [ ] Rust streaming graph builder; parallel construction; benchmark suite.
- [ ] Configure GitHub OIDC trusted publishing on PyPI.
- [ ] Documentation hosting domain and analytics endpoint (pending Pages).
- [ ] Security/dependency scanning policy completion.
- [ ] Independent scientific review; reproducibility/benchmark/limitations reports; stable v1.0 API.

## COMPLETED (framework phase additions)

### Introspection and selection (Phase 6)
- [x] `brain.info()` / `BrainInfo` with `summary()`; `brain.capabilities()` machine-readable report.
- [x] Deterministic substrate fingerprint (sha256 over CSR + body IDs); exposed as `brain.fingerprint`.
- [x] Checkpoint substrate-fingerprint validation (AXW002 on incompatible connectome).
- [x] Neuron selection API (`core/selection.py`): `all()` / `ids()` / `by_mask()` / `by_type()` / `by_region()`, order-preserving, AXW010 on unknown IDs; exposed as `brain.graph.neurons`.
- [x] Selection tables built from annotation columns at install time (`build_annotations`, alias-resolved, degraded gracefully when columns are absent).
- [x] Selections wired into all three `ConnectomeLayer` backends (`selection=` parameter, `selection_body_ids` provenance, AXW010 on non-selection argument).
- [x] Brain docs page documents info/capabilities/fingerprint/selection/selections-in-layers.
