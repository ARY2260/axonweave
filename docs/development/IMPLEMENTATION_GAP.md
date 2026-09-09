# AxonWeave Implementation Gap Analysis

Status: internal development document. Produced at the start of the 0.2
development cycle against the continuation specification.

## 1. Currently implemented (working, tested)

| Area | API | Tests |
|---|---|---|
| Substrate loading | `axonweave.load`, `SubstrateRegistry` | `test_data_pipeline.py` |
| Provisioning | `axonweave substrate install male-cns:v1.0`, resumable downloads, GCS md5 + sha256 verification | `test_data_pipeline.py` |
| Graph | `ConnectomeGraph` (CSR, body-ID mapping, npz round-trip) | `test_graph.py` |
| Sparse layers | `axonweave.torch.ConnectomeLayer`, `axonweave.keras.ConnectomeLayer`, `axonweave.numpy.ConnectomeLayer` | `test_torch_layer.py`, `test_keras_layer.py`, `test_numpy_layer.py` |
| Cross-backend equivalence | NumPy = PyTorch = Keras propagation | `test_cross_backend.py` |
| Signals | `SignalPolicy`, `NeurotransmitterGain`, `LeakyPropagation` | `test_signals.py` |
| Errors | `AXW000`–`AXW006`, `AXW010`, `AXW101` | exercised throughout |
| Docs site | React/Vite renderer, DESIGN.md token system, Prism, MathJax, search, resizable sidebar | CI `docs.yml` |

## 2. Implemented in this cycle (needs CI validation)

| Area | API | Status |
|---|---|---|
| Dynamics | `axonweave.dynamics.LIF / AdaptiveLIF / Rate / DynamicsPolicy` | implemented + local tests pending commit |
| Encoders | `axonweave.encoders.ImageEncoder / TokenEncoder / SensorEncoder` | implemented |
| Decoders | `axonweave.decoders.ActionDecoder / TokenDecoder / ClassificationHead` | implemented |
| Learning | `axonweave.learning.STDP / DopamineSTDP` (three-factor reward modulation) | implemented |
| Experiment loop | `axonweave.experiment.Agent` (encode → dynamics → decode → env → reward → plasticity), JSON-lines logging, checkpoints | implemented |
| Brain facade | `brain.task(...)`, `brain.agent(...)`, `brain.layer(...)` (alias), `brain.simulate`, `brain.experiment` | implemented |
| Torch high-level | `axonweave.frameworks.torch.BrainModel / ConnectomeBlock / Input / Readout` | implemented, requires torch (CI) |
| Docs pages | framework, dynamics, encoders, learning, experiment | added to docs-site |

## 3. Missing abstractions (planned order)

1. **Neuron selection** (`brain.neurons`, `NeuronCollection`) — blocked on substrate
   annotations schema; installer already provisions `annotations.feather`. Do not
   fabricate fields; inspect schema first.
2. **Protocols/types module** (`axonweave.types`, `axonweave.protocols`) — formal
   `TensorSpec/StateSpec/DynamicsProtocol` contracts. Current ABC-style
   `DynamicsModel` covers the minimum; formal protocols are additive later.
3. **`brain.info()` / `BrainInfo`** — trivial additive wrapper over manifest +
   graph fingerprint.
4. **`axonweave.capabilities()`** — backend/sparse/device capability reporting.
5. **Readout package** — `axonweave.readout.ClassificationReadout/TokenReadout`
   as first-class re-exports of decoders.
6. **Checkpoint substrate-fingerprint validation** — checkpoint currently stores
   substrate id but does not yet refuse a mismatched graph. Must add before
   Milestone B.
7. **Keras/JAX high-level adapters** — only after torch path is CI-green.

## 4. Missing tests

- `tests/test_dynamics.py` (added this cycle)
- `tests/test_encoders_decoders.py` (added)
- `tests/test_learning.py` (added)
- `tests/test_experiment.py` (added)
- Torch-side `BrainModel/ConnectomeBlock` tests (added, torch-gated)
- Keras-side high-level adapter tests — not started (module not implemented)
- Rust: `cargo test` has no unit tests in `rust/src/lib.rs` yet

## 5. Missing documentation

- `docs/` conceptual pages for encoders/dynamics/learning/experiment exist in
  `docs-site/content/` only; mirror to `docs/` (single-source question pending —
  spec §45 prefers docs/ as source of truth; currently content is duplicated
  by copy at build-authoring time, noted as tech debt).
- `neuron-selection.md`, `checkpoints.md`, `capabilities.md` — pages must wait
  for the corresponding APIs (no placeholder pages).

## 6. CI coverage

- `ci.yml` python matrix + backend-smoke already runs the full test suite;
  new tests are picked up automatically. No workflow changes needed for this
  cycle (spec §30: do not rewrite workflows without need).
- No new CI job required until JAX adapter exists.

## 7. Risks

- `ConnectomeBlock` multi-step spiking dynamics is NumPy-executed inside the
  torch forward (CPU round-trip). This is correct but slow and breaks
  autograd through dynamics. Documented as experimental; surrogate gradients
  are a Milestone B+ item.
- Plasticity `update()` operates on CSR data arrays; index math must be
  validated against non-square/duplicate-edge inputs (tests added).
- Local env has no torch/TF: all backend tests authored for CI, not run
  locally (spec §28/§33 — do not claim local execution).
