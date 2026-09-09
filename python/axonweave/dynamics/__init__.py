"""Neuron dynamics models (Phase 3).

Dynamics are explicit model choices applied on top of the structural
connectome. They never modify the graph itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import BiologicalAssumptionError


class DynamicsModel:
    """Base class for time-stepped neuron dynamics."""

    name = "base"

    def initial_state(self, n_neurons: int, batch_shape: tuple[int, ...]):
        raise NotImplementedError

    def step(self, state: dict[str, Any], input_current, W):
        """Advance one time step. ``W`` is the sparse weight matrix (scipy).

        Returns (output_activity, new_state).
        """
        raise NotImplementedError


@dataclass
class LIF(DynamicsModel):
    """Leaky Integrate-and-Fire with refractory period."""

    tau: float = 20.0            # membrane time constant (ms)
    v_rest: float = -65.0        # resting potential (mV)
    v_threshold: float = -50.0   # spike threshold (mV)
    v_reset: float = -70.0       # reset potential (mV)
    refractory: float = 2.0      # absolute refractory period (ms)
    dt: float = 1.0              # integration time step (ms)

    name = "lif"

    def initial_state(self, n_neurons: int, batch_shape: tuple[int, ...] = ()):
        import numpy as np

        shape = (*batch_shape, n_neurons)
        return {
            "v": np.full(shape, self.v_rest, dtype=np.float32),
            "refrac_until": np.zeros(shape, dtype=np.float32),
            "t": 0.0,
        }

    def step(self, state, input_current, W):
        import numpy as np

        t = state["t"] + self.dt
        can_spike = state["refrac_until"] <= t
        # Synaptic input through the sparse connectome.
        synaptic = input_current + (input_current @ W if W is not None else 0)
        dv = (-(state["v"] - self.v_rest) + synaptic) * (self.dt / self.tau)
        v = np.where(can_spike, state["v"] + dv, state["v"])
        spiked = can_spike & (v >= self.v_threshold)
        v = np.where(spiked, self.v_reset, v)
        new_state = {
            "v": v,
            "refrac_until": np.where(spiked, t + self.refractory, state["refrac_until"]),
            "t": t,
        }
        return spiked.astype(np.float32), new_state


@dataclass
class AdaptiveLIF(LIF):
    """LIF with an activity-dependent adaptive firing threshold."""

    tau_adapt: float = 200.0     # adaptation time constant (ms)
    delta_threshold: float = 0.5  # threshold increase per spike (mV)

    name = "adaptive_lif"

    def initial_state(self, n_neurons: int, batch_shape: tuple[int, ...] = ()):
        import numpy as np

        state = super().initial_state(n_neurons, batch_shape)
        shape = (*batch_shape, n_neurons)
        state["threshold"] = np.full(shape, self.v_threshold, dtype=np.float32)
        return state

    def step(self, state, input_current, W):
        import numpy as np

        t = state["t"] + self.dt
        can_spike = state["refrac_until"] <= t
        synaptic = input_current + (input_current @ W if W is not None else 0)
        dv = (-(state["v"] - self.v_rest) + synaptic) * (self.dt / self.tau)
        v = np.where(can_spike, state["v"] + dv, state["v"])
        spiked = can_spike & (v >= state["threshold"])
        v = np.where(spiked, self.v_reset, v)
        # Threshold relaxes toward base value, jumps up on spike.
        threshold = state["threshold"] + (self.v_threshold - state["threshold"]) * (self.dt / self.tau_adapt)
        threshold = np.where(spiked, threshold + self.delta_threshold, threshold)
        new_state = {
            "v": v,
            "threshold": threshold,
            "refrac_until": np.where(spiked, t + self.refractory, state["refrac_until"]),
            "t": t,
        }
        return spiked.astype(np.float32), new_state


@dataclass
class Rate(DynamicsModel):
    """Instantaneous rate model — smooth and cheap for gradient training."""

    gain: float = 1.0
    baseline: float = 0.0
    dt: float = 1.0

    name = "rate"

    def initial_state(self, n_neurons: int, batch_shape: tuple[int, ...] = ()):
        return {"t": 0.0}

    def step(self, state, input_current, W):
        synaptic = input_current + (input_current @ W if W is not None else 0)
        activity = self.baseline + self.gain * synaptic
        return activity, {"t": state["t"] + self.dt}


@dataclass
class DynamicsPolicy:
    """Global dynamics with optional per-neuron-type overrides.

    Overrides are keyed by neuron *type label* (from substrate annotations,
    e.g. "kenyon_cell"). Types without annotations raise an actionable error
    rather than silently falling back.
    """

    default: DynamicsModel = field(default_factory=Rate)
    overrides: dict[str, DynamicsModel] = field(default_factory=dict)

    def model_for(self, neuron_type: str | None) -> DynamicsModel:
        if neuron_type is None:
            return self.default
        if neuron_type in self.overrides:
            return self.overrides[neuron_type]
        raise BiologicalAssumptionError(
            f"AXW005: no dynamics override registered for neuron type {neuron_type!r}; "
            f"known types: {sorted(self.overrides) or 'none'}"
        )
