# Library Code Tokens and Contracts

These tokens are short identifiers used in agent instructions and code review.

| Token | Meaning |
|---|---|
| `AXW-PROV` | substrate provisioning/provenance |
| `AXW-GRAPH` | graph topology/indexing |
| `AXW-BIO` | explicit biological-model assumptions |
| `AXW-BACKEND` | framework adapter behavior |
| `AXW-DEVICE` | device capability/placement |
| `AXW-RUST` | Rust/PyO3 boundary |
| `AXW-CKPT` | checkpoint identity |
| `AXW-TEST` | test requirement |
| `AXW-DOC` | public documentation requirement |
| `AXW-API` | public API stability |

Agent rule: if a change crosses one of these areas, inspect its contract in `ARCHITECTURE.md` and add the relevant test/documentation.
