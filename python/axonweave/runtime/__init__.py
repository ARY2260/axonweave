"""Connectome runtime (Temporal Runtime Alpha): explicit stateful execution.

This package turns the existing primitives (dynamics, sparse propagation,
delays, plasticity) into one coherent temporal execution model with explicit
state, sequence semantics and deterministic replay:

    runtime.reset_state()
    for x_t in stream:
        y_t = runtime.step(x_t)

Key semantics (backend-neutral; adapters must preserve them):

- ``step(x_t)`` advances exactly one timestep and never resets state.
- ``forward_sequence(x)`` with input ``[..., T, F]`` is equivalent to calling
  ``step(x[..., t, :])`` for ``t`` in ``0..T-1`` from the same initial state,
  under deterministic execution. An equivalence test enforces this.
- ``get_state()`` returns a deep-copied, serializable state object;
  ``set_state(state)`` restores it so trajectories are reproducible and
  branchable.
- ``detach_state()`` detaches carried gradients (framework adapters) for
  truncated BPTT; on the NumPy reference path it is an explicit no-op that
  still returns self for API symmetry.

The runtime is *recurrent by construction*: state at ``t+1`` depends on state
and activity at ``t`` through the dynamics model. It is never simulated by
stacking feed-forward layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dynamics import DynamicsModel, Rate
from ..errors import ApiUsageError

__all__ = [
    "NeuronState",
    "SynapticState",
    "PlasticityState",
    "RuntimeState",
    "ConnectomeRuntime",
]


@dataclass
class NeuronState:
    """Membrane/state variables carried between timesteps.

    ``variables`` holds the dynamics model's per-neuron arrays (e.g. ``v``,
    ``refrac_until``, ``threshold``) plus the scalar ``t`` clock. Shapes and
    keys are owned by the dynamics model; the runtime treats them opaquely
    apart from deep-copying on capture.
    """

    variables: dict = field(default_factory=dict)

    def copy(self) -> "NeuronState":
        return NeuronState(variables={k: (v.copy() if hasattr(v, "copy") else v)
                                      for k, v in self.variables.items()})


@dataclass
class SynapticState:
    """Delayed signals and synaptic state carried between timesteps.

    Reserved for the delay-engine and receptor paths; the reference runtime
    keeps it empty (delayed propagation is available through the dedicated
    delay engine and composed at the adapter level).
    """

    delayed: dict = field(default_factory=dict)

    def copy(self) -> "SynapticState":
        return SynapticState(delayed={k: (v.copy() if hasattr(v, "copy") else v)
                                      for k, v in self.delayed.items()})


@dataclass
class PlasticityState:
    """Eligibility/STDP/reward traces carried between timesteps."""

    traces: dict = field(default_factory=dict)

    def copy(self) -> "PlasticityState":
        return PlasticityState(traces={k: (v.copy() if hasattr(v, "copy") else v)
                                       for k, v in self.traces.items()})


@dataclass
class RuntimeState:
    """Complete, serializable runtime snapshot for replay and branching."""

    timestep: int = 0
    neuron: NeuronState = field(default_factory=NeuronState)
    synaptic: SynapticState = field(default_factory=SynapticState)
    plasticity: PlasticityState = field(default_factory=PlasticityState)
    rng: dict = field(default_factory=dict)

    def copy(self) -> "RuntimeState":
        return RuntimeState(
            timestep=self.timestep,
            neuron=self.neuron.copy(),
            synaptic=self.synaptic.copy(),
            plasticity=self.plasticity.copy(),
            rng=dict(self.rng),
        )

    def to_dict(self) -> dict:
        """Serialize to plain Python structures (JSON-friendly scalars aside)."""

        def _plain(value):
            if isinstance(value, np.ndarray):
                return value
            if isinstance(value, (int, float, str, bool, type(None))):
                return value
            if isinstance(value, dict):
                return {k: _plain(v) for k, v in value.items()}
            return value

        return {
            "timestep": self.timestep,
            "neuron": {k: _plain(v) for k, v in self.neuron.variables.items()},
            "synaptic": {k: _plain(v) for k, v in self.synaptic.delayed.items()},
            "plasticity": {k: _plain(v) for k, v in self.plasticity.traces.items()},
            "rng": dict(self.rng),
        }


class ConnectomeRuntime:
    """Stateful temporal execution over a connectome (NumPy reference path).

    Composes the sparse substrate (``W``), a dynamics model, and explicit
    state into a single stepping object. Adapter-level runtimes (torch /
    keras / jax) wrap this with backend tensors and must preserve the
    semantics above.

    Args:
        graph: the ``ConnectomeGraph`` to execute over (its CSR weights and
            body IDs define the neuron space). May be ``None`` together with
            an explicit ``n_neurons`` for graph-free dynamics testing.
        dynamics: dynamics model instance (default ``Rate``).
        n_neurons: override the neuron-space size when no graph is given.
    """

    def __init__(self, graph=None, dynamics: DynamicsModel | None = None,
                 n_neurons: int | None = None):
        if graph is not None:
            self.weights = graph.weights
            self.body_ids = np.asarray(graph.body_ids, dtype=np.int64)
            self._n = self.weights.shape[0]
        elif n_neurons is not None:
            self.weights = None
            self.body_ids = np.arange(n_neurons, dtype=np.int64)
            self._n = int(n_neurons)
        else:
            raise ApiUsageError(
                "AXW010: ConnectomeRuntime needs a graph or an explicit n_neurons")
        self.dynamics = dynamics if dynamics is not None else Rate()
        if not isinstance(self.dynamics, DynamicsModel):
            raise ApiUsageError(
                "AXW010: dynamics must be a DynamicsModel instance; "
                f"got {type(self.dynamics).__name__}")
        self._state: RuntimeState | None = None

    # -- introspection -------------------------------------------------------

    @property
    def n_neurons(self) -> int:
        return self._n

    @property
    def timestep(self) -> int:
        return self._state.timestep if self._state is not None else 0

    def memory_estimate(self, dtype: str = "float32") -> dict:
        """Per-component memory footprint of one full state, in bytes."""
        itemsize = np.dtype(dtype).itemsize
        state = self._state or self._fresh_state(())
        neuron_bytes = sum(
            v.nbytes if isinstance(v, np.ndarray) else 0
            for v in state.neuron.variables.values()
        )
        edge_bytes = int(self.weights.nnz * itemsize) if self.weights is not None else 0
        return {
            "neuron_state": neuron_bytes,
            "edge_parameters": edge_bytes,
            "delay_buffer": 0,
            "receptor_state": 0,
            "plasticity_state": 0,
            "dtype": dtype,
        }

    # -- state management ------------------------------------------------------

    def _fresh_state(self, batch_shape: tuple[int, ...]) -> RuntimeState:
        inner = self.dynamics.initial_state(self._n, batch_shape)
        return RuntimeState(
            timestep=0,
            neuron=NeuronState(variables=dict(inner)),
        )

    def reset_state(self, batch_shape: tuple[int, ...] = ()) -> None:
        """Clear all carried state; the next step starts from t=0."""
        self._state = None
        self._batch_shape = batch_shape

    def get_state(self) -> RuntimeState:
        """Return a deep copy of the current state (safe to keep/branch)."""
        if self._state is None:
            return self._fresh_state(()).copy()
        return self._state.copy()

    def set_state(self, state: RuntimeState) -> None:
        """Restore a previously captured state (deep-copied on the way in)."""
        if not isinstance(state, RuntimeState):
            raise ApiUsageError(
                "AXW010: set_state expects a RuntimeState from get_state(); "
                f"got {type(state).__name__}")
        self._state = state.copy()

    def detach_state(self) -> "ConnectomeRuntime":
        """Detach carried gradients (truncated BPTT support).

        NumPy reference path: no-op returning self. Framework adapters
        override this to break autograd graphs across sequence chunks.
        """
        return self

    # -- execution ---------------------------------------------------------------

    def step(self, x_t) -> np.ndarray:
        """Advance exactly one timestep. State persists between calls."""
        x = np.asarray(x_t, dtype=np.float32)
        if x.shape[-1] != self._n:
            raise ApiUsageError(
                f"AXW010: expected last dimension {self._n}, got {x.shape[-1]}")
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        if self._state is None:
            self._state = self._fresh_state(x.shape[:-1])
        inner = dict(self._state.neuron.variables)
        activity, inner_new = self.dynamics.step(inner, x, self.weights)
        inner_new["t"] = inner.get("t", 0.0) + getattr(self.dynamics, "dt", 1.0)
        self._state.neuron = NeuronState(variables=inner_new)
        self._state.timestep += 1
        out = np.asarray(activity, dtype=np.float32)
        return out.reshape(x.shape[0], -1).squeeze(0) if squeeze else out.reshape(*x.shape[:-1], -1)

    def forward_sequence(self, x) -> np.ndarray:
        """Run a sequence ``[..., T, F]``; returns ``[..., T, n_neurons]``.

        Equivalent to sequential :meth:`step` calls from the same initial
        state under deterministic execution — enforced by tests.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim < 2 or x.shape[-1] != self._n:
            raise ApiUsageError(
                f"AXW010: forward_sequence expects [..., T, {self._n}] (last dim "
                f"is the feature/neuron axis), got shape {x.shape}")
        lead = x.shape[:-2]
        T = x.shape[-2]
        xf = x.reshape(-1, T, self._n)
        outs = []
        self.reset_state(batch_shape=lead)
        for t in range(T):
            outs.append(self.step(xf[:, t, :]))
        return np.stack(outs, axis=1).reshape(*lead, T, self._n)
