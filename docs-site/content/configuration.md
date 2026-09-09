# Configuration

## Cache location

Set `AXONWEAVE_HOME` to control the local substrate cache:

```bash
export AXONWEAVE_HOME=/data/axonweave
```

On Windows PowerShell:

```powershell
$env:AXONWEAVE_HOME = "D:\axonweave"
```

## Biological assumptions

Keep receptor/sign, delay, dynamics and plasticity policies in version-controlled experiment configuration. A trained checkpoint should record these policy identities.
