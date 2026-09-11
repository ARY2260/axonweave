"""Neuron dynamics models (Phase 3).

Dynamics are explicit model choices applied on top of the structural
connectome. They never modify the graph itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import BiologicalAssumptionError

from .surrogate import (
    ATanSurrogate,
    PiecewiseSurrogate,
    SigmoidSurrogate,
    StraightThroughEstimator,
    SurrogateGradient,
)


def _propagate(input_current, W):
    """Propagate input currents through the sparse connectome (pre -> post).

    Returns ``None`` when there is no weight matrix (``W is None``).
    """
    if W is None:
        return None
    import numpy as np

    from .. import native as _native

    x = np.asarray(input_current, dtype=np.float32)
    if x.ndim == 1:
        return _native.sparse_matmul_transpose(W, x)
    return _native.csr_matmul_2d_transpose(W, x)


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

        from .. import native as _native

        t = state["t"] + self.dt
        synaptic = _propagate(input_current, W)
        if synaptic is None:
            current = np.zeros_like(state["v"], dtype=np.float32)
        else:
            current = np.asarray(synaptic, dtype=np.float32)
        shape = state["v"].shape
        spikes, v, refrac_until = _native.lif_step(
            state["v"], state["refrac_until"], current, state["t"],
            self.tau, self.v_rest, self.v_threshold, self.v_reset,
            self.refractory, self.dt,
        )
        new_state = {
            "v": v.reshape(shape),
            "refrac_until": refrac_until.reshape(shape),
            "t": t,
        }
        return spikes.reshape(shape), new_state


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

        from .. import native as _native

        t = state["t"] + self.dt
        synaptic = _propagate(input_current, W)
        if synaptic is None:
            current = np.zeros_like(state["v"], dtype=np.float32)
        else:
            current = np.asarray(synaptic, dtype=np.float32)
        shape = state["v"].shape
        spikes, v, threshold, refrac_until = _native.adaptive_lif_step(
            state["v"], state["refrac_until"], state["threshold"], current,
            state["t"], self.tau, self.v_rest, self.v_threshold, self.v_reset,
            self.refractory, self.tau_adapt, self.delta_threshold, self.dt,
        )
        new_state = {
            "v": v.reshape(shape),
            "threshold": threshold.reshape(shape),
            "refrac_until": refrac_until.reshape(shape),
            "t": t,
        }
        return spikes.reshape(shape), new_state


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
        import numpy as np

        from .. import native as _native

        # Activity propagates along directed edges: pre -> post (x @ W).
        if W is not None:
            synaptic = _propagate(input_current, W)
        else:
            synaptic = input_current
        syn = np.asarray(synaptic, dtype=np.float32)
        activity = _native.rate_step(syn.ravel(), self.gain, self.baseline)
        return activity.reshape(syn.shape), {"t": state["t"] + self.dt}


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


from .spiking import SurrogateAdaptiveLIF, SurrogateLIF  # noqa: E402

__all__ = [
    "ATanSurrogate",
    "AdaptiveLIF",
    "DynamicsModel",
    "DynamicsPolicy",
    "LIF",
    "PiecewiseSurrogate",
    "Rate",
    "SigmoidSurrogate",
    "StraightThroughEstimator",
    "SurrogateAdaptiveLIF",
    "SurrogateGradient",
    "SurrogateLIF",
]
