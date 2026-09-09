# Neuron Dynamics

AxonWeave separates the **static connectome** (structural wiring) from **neuron dynamics** (how state evolves). Dynamics are explicit, configurable model choices.

## Available models

### Leaky Integrate-and-Fire (LIF)

Membrane potential with spike generation:

$$\tau \frac{dV}{dt} = -(V - V_{rest}) + \sum_j w_{ij} \, s_j(t)$$

```python
from axonweave.dynamics import LIF

dyn = LIF(
    tau=20.0,          # membrane time constant (ms)
    v_rest=-65.0,      # resting potential (mV)
    v_threshold=-50.0, # spike threshold (mV)
    v_reset=-70.0,     # reset potential after a spike
    refractory=2.0,    # absolute refractory period (ms)
)
```

### Adaptive LIF

Adds a dynamic firing threshold that increases after each spike:

```python
from axonweave.dynamics import AdaptiveLIF

dyn = AdaptiveLIF(tau=20.0, tau_adapt=200.0, delta_threshold=0.5)
```

### Rate-based approximation

Instantaneous firing-rate model — cheaper and smoother for gradient-based training:

```python
from axonweave.dynamics import Rate

dyn = Rate(gain=1.0, baseline=0.0)
```

## Configuration

Override dynamics globally or per neuron type through a policy:

```python
from axonweave.dynamics import DynamicsPolicy

policy = DynamicsPolicy(
    default=LIF(tau=20.0),
    overrides={"kenyon_cell": Rate(gain=2.0)},
)
```

## Time-stepped propagation

Dynamics run over discrete time steps on top of the sparse propagation:

```text
input currents → weighted sum (connectome) → state update (dynamics) → spikes/rates → next step
```

:::DOC-NOTE
The LIF equation above is a modeling choice applied to the structural graph — the upstream dataset provides connectivity and synaptic counts, not membrane parameters. Record your dynamics configuration in experiment metadata.
:::

## Determinism

All dynamics models support a deterministic simulation mode: seed the state RNG explicitly for reproducible runs (`BrainModel(..., seed=0)`).
