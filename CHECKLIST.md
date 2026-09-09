# AxonWeave Implementation Checklist

## Completed in this package

- [x] Complete repository README.
- [x] Architecture and AI-agent governance documents.
- [x] Frontend design constraints.
- [x] Separate web/library agent token definitions.
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
- [x] Python 3.10–3.14 CI matrix definition.
- [x] Ubuntu/macOS/Windows CI matrix definition.
- [x] Rust/PyO3 CI.
- [x] Wheel build and PyPI trusted-publishing workflow.
- [x] Source-distribution build check.
- [x] `.gitignore` covering agents, caches, builds and secrets.

## Remaining before claiming production release

- [ ] Run all CI jobs on GitHub and fix runner-specific failures.
- [ ] Verify Python 3.14 compatibility for every dependency/backend.
- [ ] Add GPU/TPU self-hosted or vendor runners and execute sparse-kernel tests.
- [ ] Validate exact upstream schemas and release fixtures.
- [ ] Add stable checksums for all upstream files.
- [ ] Replace in-memory graph assembly with disk-backed/streaming build.
- [ ] Implement full receptor/dynamics/delay/plasticity model.
- [ ] Implement portable `.awb` artifact.
- [ ] Add API-doc generation.
- [ ] Configure GitHub OIDC trusted publishing on PyPI.
- [ ] Configure documentation hosting domain and analytics endpoint.
- [ ] Complete security/dependency scanning policy.
- [x] GitHub Pages deployment workflow for the docs site.
- [x] SPA fallback (`404.html`) for direct documentation routes on static hosting.
