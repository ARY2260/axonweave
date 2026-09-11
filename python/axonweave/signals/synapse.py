from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from .. import native as _native
from ..errors import _require


class NeurotransmitterType(enum.Enum):
    GLUTAMATE = "glutamate"
    GABA = "gaba"
    ACETYLCHOLINE = "acetylcholine"
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    OCTOPAMINE = "octopamine"


@dataclass(frozen=True)
class NeurotransmitterProperties:
    name: str
    sign: int
    decay_ms: float
    vesicle_release_probability: float
    reuptake_rate: float


SYNAPSE_PROPERTIES: dict[NeurotransmitterType, NeurotransmitterProperties] = {
    NeurotransmitterType.GLUTAMATE: NeurotransmitterProperties(
        name="glutamate",
        sign=1,
        decay_ms=2.0,
        vesicle_release_probability=0.75,
        reuptake_rate=0.90,
    ),
    NeurotransmitterType.GABA: NeurotransmitterProperties(
        name="gaba",
        sign=-1,
        decay_ms=5.0,
        vesicle_release_probability=0.65,
        reuptake_rate=0.85,
    ),
    NeurotransmitterType.ACETYLCHOLINE: NeurotransmitterProperties(
        name="acetylcholine",
        sign=1,
        decay_ms=3.0,
        vesicle_release_probability=0.70,
        reuptake_rate=0.80,
    ),
    NeurotransmitterType.DOPAMINE: NeurotransmitterProperties(
        name="dopamine",
        sign=1,
        decay_ms=50.0,
        vesicle_release_probability=0.50,
        reuptake_rate=0.60,
    ),
    NeurotransmitterType.SEROTONIN: NeurotransmitterProperties(
        name="serotonin",
        sign=1,
        decay_ms=80.0,
        vesicle_release_probability=0.45,
        reuptake_rate=0.55,
    ),
    NeurotransmitterType.OCTOPAMINE: NeurotransmitterProperties(
        name="octopamine",
        sign=1,
        decay_ms=40.0,
        vesicle_release_probability=0.40,
        reuptake_rate=0.50,
    ),
}


class SynapseNeurotransmitterModel:
    def __init__(
        self,
        n_synapses: int,
        nt_types: np.ndarray | None = None,
        rng_seed: int = 0,
    ) -> None:
        _require(n_synapses > 0, f"n_synapses must be > 0, got {n_synapses}")

        self._n_synapses = n_synapses
        self._rng = np.random.default_rng(rng_seed)

        if nt_types is None:
            self._nt_types = np.array(
                [NeurotransmitterType.GLUTAMATE] * n_synapses, dtype=object
            )
        else:
            _require(
                nt_types.shape == (n_synapses,),
                f"nt_types must have shape ({n_synapses},), got {nt_types.shape}",
            )
            self._nt_types = nt_types

        self._signs = np.array(
            [SYNAPSE_PROPERTIES[nt].sign for nt in self._nt_types], dtype=np.float64
        )
        self._release_probs = np.array(
            [SYNAPSE_PROPERTIES[nt].vesicle_release_probability for nt in self._nt_types],
            dtype=np.float64,
        )
        self._decay_rates = np.array(
            [SYNAPSE_PROPERTIES[nt].reuptake_rate for nt in self._nt_types],
            dtype=np.float64,
        )

        self._vesicle_pool = np.ones(n_synapses, dtype=np.float64)
        self._concentration = np.zeros(n_synapses, dtype=np.float64)

    def step(
        self,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        W: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, dict]:
        _require(
            pre_activity.shape == (self._n_synapses,),
            f"pre_activity must have shape ({self._n_synapses},), got {pre_activity.shape}",
        )
        _require(
            post_activity.shape == (self._n_synapses,),
            f"post_activity must have shape ({self._n_synapses},), got {post_activity.shape}",
        )
        _require(
            W.shape == (self._n_synapses,),
            f"W must have shape ({self._n_synapses},), got {W.shape}",
        )

        stochastic = self._rng.random(self._n_synapses)
        concentration, vesicle_pool, per_synapse_current, post_current = _native.vesicle_release_step(
            self._vesicle_pool, self._concentration, pre_activity,
            self._release_probs, self._decay_rates, stochastic,
            self._signs, W,
        )
        self._concentration = concentration
        self._vesicle_pool = vesicle_pool

        state_dict = {
            "vesicle_pool": self._vesicle_pool.copy(),
            "concentration": self._concentration.copy(),
            "per_synapse_current": per_synapse_current.copy(),
        }

        return post_current, state_dict

    def get_receptor_currents(
        self, pre_activity: np.ndarray, W: np.ndarray
    ) -> np.ndarray:
        _require(
            pre_activity.shape == (self._n_synapses,),
            f"pre_activity must have shape ({self._n_synapses},), got {pre_activity.shape}",
        )
        _require(
            W.shape == (self._n_synapses,),
            f"W must have shape ({self._n_synapses},), got {W.shape}",
        )
        return _native.nt_currents(self._signs, W, pre_activity)

    def get_state(self) -> dict:
        return {
            "vesicle_pool": self._vesicle_pool.copy(),
            "concentration": self._concentration.copy(),
        }

    def set_state(self, state: dict) -> None:
        _require(
            "vesicle_pool" in state and "concentration" in state,
            "state must contain 'vesicle_pool' and 'concentration' keys",
        )
        _require(
            state["vesicle_pool"].shape == (self._n_synapses,),
            f"vesicle_pool must have shape ({self._n_synapses},)",
        )
        _require(
            state["concentration"].shape == (self._n_synapses,),
            f"concentration must have shape ({self._n_synapses},)",
        )
        self._vesicle_pool = state["vesicle_pool"].copy()
        self._concentration = state["concentration"].copy()
