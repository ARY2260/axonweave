# Troubleshooting

## `AXW001` substrate not installed

Run:

```bash
axonweave substrate install male-cns:v1.0
```

## Download interrupted

The installer uses `.part` files and resumes with HTTP Range requests when the source server supports them. It is safe to rerun the command.

## Sparse operation fails on an accelerator

Check the installed framework version and whether its sparse operator is implemented on the selected device. AxonWeave does not silently fall back to CPU.

## Memory pressure

Use a smaller selected subgraph, lower batch size, or a backend/device with sufficient memory. Do not convert the complete graph to a dense matrix.
