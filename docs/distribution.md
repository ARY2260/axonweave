# Substrate Distribution

The Python wheel and the biological substrate are separate artifacts.

## Online

```bash
pip install axonweave
axonweave substrate install male-cns:v1.0
```

## Cache

The default cache is under the platform's user cache directory and can be redirected with `AXONWEAVE_HOME`.

## Offline deployment

A future `.awb` pack format is planned. The intended workflow is:

```bash
axonweave substrate pack male-cns:v1.0 --output male-cns-v1.0.awb
```

then on an isolated machine:

```bash
axonweave substrate install ./male-cns-v1.0.awb
```

## Why not bundle it in PyPI?

The upstream release includes files ranging from tens of megabytes to multi-gigabyte synapse resources. A normal Python wheel should contain software, not force every installation to transfer the entire biological archive.
