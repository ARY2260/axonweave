"""Learning modes and plasticity rules (Phase 4).

Modes 1-3 and 5 use framework optimizers (backpropagation). Mode 4 uses
local plasticity rules updated from pre/post activity and reward signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import native as _native

import numpy as np


@dataclass
class STDP:
    """Spike-timing-dependent plasticity (pairwise, exponential traces)."""

    a_plus: float = 0.01     # LTP amplitude
    a_minus: float = 0.012   # LTD amplitude
    tau_pre: float = 20.0    # pre-synaptic trace time constant (ms)
    tau_post: float = 20.0   # post-synaptic trace time constant (ms)
    w_min: float | None = None
    w_max: float | None = None

    def initial_traces(self, n_neurons: int):
        return {
            "pre": np.zeros(n_neurons, dtype=np.float32),
            "post": np.zeros(n_neurons, dtype=np.float32),
        }

    def update(self, W, state, pre_activity, post_activity, dt: float = 1.0, reward: float | None = None):
        """Apply one plasticity step to the CSR data of ``W``.

        ``state`` carries persistent traces. Returns the updated weight data.
        Reward (if given) modulates the update (three-factor rule).
        """
        data, pre_new, post_new = _native.stdp_update(
            W, state["pre"], state["post"], pre_activity, post_activity,
            self.a_plus, self.a_minus, self.tau_pre, self.tau_post, dt,
            reward=reward, w_min=self.w_min, w_max=self.w_max,
        )
        state["pre"] = pre_new
        state["post"] = post_new
        if W is not None and W.nnz:
            W.data = data
        return W.data if W is not None else None


@dataclass
class DopamineSTDP:
    """Three-factor, reward-modulated STDP (dopamine signal)."""

    base: STDP = field(default_factory=STDP)
    dopamine_gain: float = 1.0

    def initial_traces(self, n_neurons: int):
        return self.base.initial_traces(n_neurons)

    def update(self, W, state, pre_activity, post_activity, dt: float = 1.0, reward: float = 0.0):
        return self.base.update(W, state, pre_activity, post_activity, dt=dt,
                                reward=self.dopamine_gain * reward)


LEARNING_RULES = {
    "stdp": STDP,
    "dopamine_stdp": DopamineSTDP,
}
