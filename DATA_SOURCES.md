# Data Sources and Provenance

## Default substrate

**MaleCNS v1.0**, the complete Drosophila male central nervous system connectome released by Janelia/FlyEM.

Official project: `https://male-cns.janelia.org/`
Official downloads: `https://male-cns.janelia.org/download/`

The release page identifies the full connection graph as `connectome-weights-male-cns-v1.0-minconf-0.5.feather` and provides annotation, neurotransmitter, statistics, synapse and skeleton resources.

## Files used by AxonWeave

| Logical resource | Upstream file | Approx. upstream size |
|---|---|---:|
| connectivity | `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1.1 GB |
| annotations | `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 13 MB |
| neurotransmitters | `body-neurotransmitters-male-cns-v1.0.feather` | 42 MB |
| statistics | `body-stats-male-cns-v1.0-minconf-0.5.feather` | 780 MB |
| synapse points | `syn-points-male-cns-v1.0-minconf-0.5.feather` | 12.7 GB |
| synapse partners | `syn-partners-male-cns-v1.0-minconf-0.5.feather` | 6.8 GB |
| T-bar neurotransmitters | `tbar-neurotransmitters-male-cns-v1.0.feather` | 2.7 GB |

These sizes are release-page guidance, not integrity hashes. AxonWeave should add upstream checksums to the registry when the source publisher exposes stable checksums.

## Provenance requirements

Every provisioned substrate should retain:

- substrate identifier and release version;
- source URL(s);
- source filenames;
- source file sizes observed during download;
- schema fingerprint;
- graph fingerprint;
- builder version;
- AxonWeave version;
- build timestamp;
- biological-data license.

## Scientific caution

Source facts must remain distinct from model assumptions. In particular, predicted neurotransmitter labels should be preserved as data. A receptor/sign model is a user-selected computational assumption unless explicitly supported by the source literature.
