# AxonWeave Development Plan

Status markers: `[x]` done, `[~]` partially implemented (see CHECKLIST.md for
what is written vs CI-verified), `[ ]` not started.

## Phase 0 — Repository foundation

- [x] Python package structure.
- [x] PyO3/Rust boundary.
- [x] NumPy/SciPy reference layer.
- [x] PyTorch layer.
- [x] TensorFlow/Keras layer.
- [x] Substrate registry concept.
- [x] Documentation application scaffold.
- [x] CI/CD scaffold.
- [x] Agent specifications (`AGENTS.md`, docs-site `AGENTS.md`, `CODE_TOKENS.md`).

## Phase 1 — Reproducible substrate provisioning

- [x] Official MaleCNS v1.0 registry metadata.
- [x] Resumable downloads.
- [x] Local cache.
- [x] Stable upstream checksum registry (GCS-published MD5 per file, verified at install; sha256 slot ready).
- [ ] Full schema fingerprint validation against release fixtures.
- [ ] Disk-backed graph construction without retaining all edges in RAM.
- [ ] `.awb` portable substrate pack/unpack.
- [ ] Graph fingerprint and migration/versioning.

## Phase 2 — Biological model core

- [ ] Typed neuron metadata API.
- [ ] ROI/body-ID selectors (`brain.neurons` collection).
- [ ] Receptor model interface.
- [x] Neuron-type dynamics interface (`DynamicsPolicy` overrides per type).
- [ ] Synaptic delay engine.
- [ ] Synapse-level neurotransmitter model.
- [x] Plasticity API (`STDP`, `DopamineSTDP` three-factor rules).
- [x] Deterministic simulation mode (seeded state, deterministic step).
- [ ] Scientific validation fixtures.

## Phase 3 — Framework integration

- [x] PyTorch.
- [x] TensorFlow/Keras.
- [x] NumPy/SciPy.
- [ ] JAX adapter where sparse semantics are stable and useful.
- [x] Cross-backend numerical equivalence tests (NumPy = PyTorch = Keras).
- [ ] PyTorch CUDA/MPS/XPU test lanes where runners are available.
- [ ] TensorFlow GPU/TPU test lanes where runners are available.
- [ ] Mixed precision policy.
- [ ] Distributed graph partitioning.

## Phase 4 — Native performance

- [x] PyO3 boundary.
- [ ] Rust Arrow/Parquet/Feather streaming graph builder.
- [ ] Parallel graph construction.
- [ ] Sparse propagation kernels.
- [ ] Profiling/benchmark suite.
- [ ] Memory-budgeted execution.

## Phase 5 — Documentation and developer ecosystem

- [x] Browser-rendered Markdown docs.
- [x] Sidebar/navigation.
- [x] Search-ready page model.
- [x] Theme switching.
- [x] Copyable code blocks.
- [x] TOC.
- [x] Legal pages.
- [x] SEO files.
- [ ] API reference generated from Python docstrings.
- [ ] Versioned documentation.
- [~] Hosted docs deployment (Pages workflow pushed; awaiting repo visibility/Pages enablement).

## Phase 6 — Connectome computing framework (new)

High-level abstractions above the substrate/layer APIs. Additive; all
low-level APIs remain unchanged.

- [x] Neuron dynamics models: `LIF`, `AdaptiveLIF`, `Rate` (`axonweave.dynamics`).
- [x] Encoders: `ImageEncoder`, `TokenEncoder`, `SensorEncoder` (`axonweave.encoders`).
- [x] Decoders/readouts: `ActionDecoder`, `TokenDecoder`, `ClassificationHead` (`axonweave.decoders`).
- [x] Learning/plasticity rules: `STDP`, `DopamineSTDP` (`axonweave.learning`).
- [x] Experiment loop: `Agent` (encode → dynamics → decode → env → reward → plasticity), JSONL logging, checkpoints (`axonweave.experiment`).
- [~] Brain facades: `brain.task(...)`, `brain.agent(...)`, `brain.layer(...)` alias, `brain.simulate`, `brain.experiment` (agent path locally verified; task path requires torch, CI-verified pending).
- [~] Torch high-level adapters: `BrainModel`, `ConnectomeBlock`, `Input`, `Readout` (written + tests; CI-verified pending).
- [ ] Neuron selection API (`brain.neurons`, `NeuronCollection`) — blocked on substrate annotations schema inspection.
- [ ] `brain.info()` / `BrainInfo` structured metadata.
- [ ] `axonweave.capabilities()` backend/device reporting.
- [ ] First-class `axonweave.readout` package re-exports.
- [ ] Checkpoint substrate-fingerprint validation (refuse mismatched graphs).
- [ ] Keras high-level adapter (`axonweave.keras.BrainLayer`).
- [ ] JAX high-level adapter.
- [ ] Surrogate-gradient training through spiking dynamics.

## Phase 7 — Scientific release

- [ ] Independent scientific review.
- [ ] Reproducibility report.
- [ ] Benchmark report.
- [ ] Biological-model limitations report.
- [ ] Stable v1.0 API.
