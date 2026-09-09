# Interoperability

AxonWeave is intended to complement existing neuroscience tooling.

## neuPrint

Use `neuprint-python` for programmatic queries against neuPrint services. AxonWeave can ingest or reconcile identifiers through explicit adapters rather than replacing neuPrint.

## navis

Use `navis` for morphology, skeletons, meshes, visualization and morphology analysis. AxonWeave's graph state can be connected to morphology metadata through body IDs.

## Principle

Keep external scientific tools authoritative for the operations they already perform well. AxonWeave is the model/substrate layer, not a replacement for every neuroscience data-analysis package.
