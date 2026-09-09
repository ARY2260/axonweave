# API Reference

## `axonweave.load`

```python
brain = axonweave.load("male-cns:v1.0")
```

Loads an installed substrate. It does not silently download multi-gigabyte data. Provisioning is explicit through the CLI.

## `ConnectomeGraph`

```python
graph.n_neurons
graph.n_edges
graph.index(body_id)
graph.neighbors(body_id)
```

The graph preserves original body IDs and stores directed connectivity as a SciPy CSR matrix.

## `BiologicalBrain`

```python
brain.graph
brain.n_neurons
brain.torch_layer(...)
brain.keras_layer(...)
```

The object binds the graph to optional source metadata paths and framework adapters.

## `SignalPolicy`

```python
SignalPolicy(mapping={"label": 1.0}, default_gain=0.0)
```

This is an explicit model assumption. It is not an upstream biological truth.

## Error codes

`AXW001` substrate missing; `AXW002` integrity/provenance mismatch; `AXW003` schema error; `AXW004` unsupported device; `AXW005` biological assumption error; `AXW006` optional backend unavailable.
