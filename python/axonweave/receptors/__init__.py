from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import native as _native
from ..errors import ApiUsageError, BiologicalAssumptionError


class ReceptorModel:
    name = "base"
    sign: int = 0
    decay_time_constant: float = 0.0
    reverse_potential: float = 0.0
    kind = -1
    gain = 1.0

    def initial_state(self, n_synapses: int) -> dict[str, Any]:
        raise NotImplementedError

    def step(
        self,
        state: dict[str, Any],
        pre_activity: Any,
        dt: float,
    ) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError

    def _step_kernel(self, state, pre_activity, dt, voltage=None):
        """Shared native receptor kernel dispatch."""
        current, g_new = _native.receptor_step(
            state["g"], pre_activity, self.kind, self.decay_time_constant,
            self.reverse_potential, self.gain, float(self.sign),
            getattr(self, "mg_concentration", 1.0),
            getattr(self, "mg_slope", 0.08),
            getattr(self, "mg_offset", -65.0),
            voltage=voltage,
            dt=dt,
        )
        return current, g_new


@dataclass
class AMPAReceptor(ReceptorModel):
    name = "ampa"
    kind = 0
    sign: int = 1
    decay_time_constant: float = 2.0
    reverse_potential: float = 0.0
    peak_conductance: float = 0.5

    def initial_state(self, n_synapses: int) -> dict[str, Any]:
        import numpy as np
        return {"g": np.zeros(n_synapses, dtype=np.float32)}

    def step(self, state, pre_activity, dt):
        current, g_new = self._step_kernel(state, pre_activity, dt)
        return current, {"g": g_new}


@dataclass
class GABAReceptor(ReceptorModel):
    name = "gaba"
    kind = 1
    sign: int = -1
    decay_time_constant: float = 6.0
    reverse_potential: float = -70.0
    peak_conductance: float = 0.5

    def initial_state(self, n_synapses: int) -> dict[str, Any]:
        import numpy as np
        return {"g": np.zeros(n_synapses, dtype=np.float32)}

    def step(self, state, pre_activity, dt):
        current, g_new = self._step_kernel(state, pre_activity, dt)
        return current, {"g": g_new}


@dataclass
class NMDAReceptor(ReceptorModel):
    name = "nmda"
    kind = 2
    sign: int = 1
    decay_time_constant: float = 100.0
    reverse_potential: float = 0.0
    mg_concentration: float = 1.0
    mg_slope: float = 0.08
    mg_offset: float = -65.0
    peak_conductance: float = 0.5

    def _mg_block(self, voltage):
        import numpy as np
        return 1.0 / (1.0 + self.mg_concentration * np.exp(-self.mg_slope * (voltage - self.mg_offset)))

    def initial_state(self, n_synapses: int) -> dict[str, Any]:
        import numpy as np
        return {"g": np.zeros(n_synapses, dtype=np.float32)}

    def step(self, state, pre_activity, dt):
        voltage = state.get("v", -65.0)
        current, g_new = self._step_kernel(state, pre_activity, dt, voltage=voltage)
        return current, {"g": g_new, "v": voltage}


@dataclass
class DopamineReceptor(ReceptorModel):
    name = "dopamine"
    kind = 3
    sign: int = 0
    decay_time_constant: float = 500.0
    reverse_potential: float = 0.0
    subtype: str = "D1"
    gain: float = 1.0

    def __post_init__(self):
        if self.subtype not in ("D1", "D2"):
            raise ApiUsageError(
                f"AXW010: DopamineReceptor subtype must be 'D1' or 'D2', got {self.subtype!r}"
            )

    def initial_state(self, n_synapses: int) -> dict[str, Any]:
        import numpy as np
        return {"g": np.zeros(n_synapses, dtype=np.float32)}

    def step(self, state, pre_activity, dt):
        current, g_new = self._step_kernel(state, pre_activity, dt)
        return current, {"g": g_new}


@dataclass
class ReceptorPolicy:
    default: ReceptorModel = field(default_factory=AMPAReceptor)
    mapping: dict[str, ReceptorModel] = field(default_factory=dict)

    def model_for(self, neurotransmitter: str | None) -> ReceptorModel:
        if neurotransmitter is None:
            return self.default
        if neurotransmitter in self.mapping:
            return self.mapping[neurotransmitter]
        raise BiologicalAssumptionError(
            f"AXW005: no receptor model registered for neurotransmitter {neurotransmitter!r}; "
            f"known types: {sorted(self.mapping) or 'none'}"
        )
