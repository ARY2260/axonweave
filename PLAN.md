# AxonWeave Development Plan

## Phase 0 — Repository foundation

- [x] Python package structure.
- [x] PyO3/Rust boundary.
- [x] NumPy/SciPy reference layer.
- [x] PyTorch layer.
- [x] TensorFlow/Keras layer.
- [x] Substrate registry concept.
- [x] Documentation application scaffold.
- [x] CI/CD scaffold.
- [x] Agent specifications.

## Phase 1 — Reproducible substrate provisioning

- [x] Official MaleCNS v1.0 registry metadata.
- [x] Resumable downloads.
- [x] Local cache.
- [ ] Stable upstream checksum registry.
- [ ] Full schema fingerprint validation against release fixtures.
- [ ] Disk-backed graph construction without retaining all edges in RAM.
- [ ] `.awb` portable substrate pack/unpack.
- [ ] Graph fingerprint and migration/versioning.

## Phase 2 — Biological model core

- [ ] Typed neuron metadata API.
- [ ] ROI/body-ID selectors.
- [ ] Receptor model interface.
- [ ] Neuron-type dynamics interface.
- [ ] Synaptic delay engine.
- [ ] Synapse-level neurotransmitter model.
- [ ] Plasticity API.
- [ ] Deterministic simulation mode.
- [ ] Scientific validation fixtures.

## Phase 3 — Framework integration

- [x] PyTorch.
- [x] TensorFlow/Keras.
- [x] NumPy/SciPy.
- [ ] JAX adapter where sparse semantics are stable and useful.
- [ ] Cross-backend numerical equivalence tests.
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
- [ ] Hosted docs deployment.

## Phase 6 — Scientific release

- [ ] Independent scientific review.
- [ ] Reproducibility report.
- [ ] Benchmark report.
- [ ] Biological-model limitations report.
- [ ] Stable v1.0 API.
