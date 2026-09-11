from __future__ import annotations

from dataclasses import dataclass, field

from .. import native as _native
from . import AdaptiveLIF, LIF
from .surrogate import SigmoidSurrogate, SurrogateGradient


@dataclass
class SurrogateLIF(LIF):
    """LIF whose spike function carries a surrogate gradient.

    The hard spike decision is still emitted downstream by
    ``surrogate.forward``; ``surrogate.backward`` is evaluated on the
    pre-reset membrane potential and stored with the state so frameworks
    can route gradient information through the spiking path.
    """

    surrogate: SurrogateGradient = field(default_factory=SigmoidSurrogate)

    name = "surrogate_lif"

    def __post_init__(self):
        self._spike_gradient = None

    @property
    def spike_gradient(self):
        return self._spike_gradient

    def step(self, state, input_current, W):
        import numpy as np

        from . import _propagate

        t = state["t"] + self.dt
        synaptic = _propagate(input_current, W)
        if synaptic is None:
            current = np.zeros_like(state["v"], dtype=np.float32)
        else:
            current = np.asarray(synaptic, dtype=np.float32)
        shape = state["v"].shape
        spikes, v, refrac_until, gradient = _native.surrogate_lif_step(
            state["v"], state["refrac_until"], current, state["t"],
            self.tau, self.v_rest, self.v_threshold, self.v_reset,
            self.refractory, self.dt,
            self.surrogate.kind, self.surrogate.k, self.surrogate.width,
        )
        new_state = {
            "v": v.reshape(shape),
            "refrac_until": refrac_until.reshape(shape),
            "t": t,
            "spike_gradient": gradient.reshape(shape),
        }
        self._spike_gradient = gradient.reshape(shape)
        return spikes.reshape(shape), new_state


@dataclass
class SurrogateAdaptiveLIF(AdaptiveLIF):
    """Adaptive LIF whose spike function carries a surrogate gradient."""

    surrogate: SurrogateGradient = field(default_factory=SigmoidSurrogate)

    name = "surrogate_adaptive_lif"

    def __post_init__(self):
        self._spike_gradient = None

    @property
    def spike_gradient(self):
        return self._spike_gradient

    def step(self, state, input_current, W):
        import numpy as np

        from . import _propagate

        t = state["t"] + self.dt
        synaptic = _propagate(input_current, W)
        if synaptic is None:
            current = np.zeros_like(state["v"], dtype=np.float32)
        else:
            current = np.asarray(synaptic, dtype=np.float32)
        shape = state["v"].shape
        spikes, v, threshold, refrac_until, gradient = _native.surrogate_adaptive_lif_step(
            state["v"], state["refrac_until"], state["threshold"], current,
            state["t"], self.tau, self.v_rest, self.v_threshold, self.v_reset,
            self.refractory, self.tau_adapt, self.delta_threshold, self.dt,
            self.surrogate.kind, self.surrogate.k, self.surrogate.width,
        )
        new_state = {
            "v": v.reshape(shape),
            "threshold": threshold.reshape(shape),
            "refrac_until": refrac_until.reshape(shape),
            "t": t,
            "spike_gradient": gradient.reshape(shape),
        }
        self._spike_gradient = gradient.reshape(shape)
        return spikes.reshape(shape), new_state