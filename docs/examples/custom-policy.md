# Custom Signal Policy

Biological sign is an explicit modeling decision.

```python
from axonweave.signals import SignalPolicy

policy = SignalPolicy(
    mapping={
        "acetylcholine": 1.0,
        "gaba": -1.0,
    },
    default_gain=0.0,
)
```

Record the policy in experiment configuration and checkpoints. Do not describe a user-defined mapping as a source-dataset fact.
