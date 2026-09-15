"""JAX framework adapter: ConnectomeBlock and BrainModel (Phase 3).

The substrate provides sparse structure; dynamics and interfaces are
explicit model choices. Tensor execution stays entirely in JAX.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:
    from ..errors import BackendUnavailableError

    raise BackendUnavailableError(
        "AXW006: axonweave.jax requires JAX; install axonweave[jax]"
    ) from e

from ..core.selection import NeuronSelection
from ..dynamics import AdaptiveLIF, LIF, Rate, DynamicsModel
from ..errors import ApiUsageError
from .layer import ConnectomeLayer


DYNAMICS: dict[str, type[DynamicsModel]] = {
    "lif": LIF,
    "adaptive_lif": AdaptiveLIF,
    "rate": Rate,
}


def resolve_dynamics(dynamics: str | DynamicsModel | None) -> DynamicsModel:
    if dynamics is None:
        return Rate()
    if isinstance(dynamics, DynamicsModel):
        return dynamics
    try:
        return DYNAMICS[dynamics]()
    except KeyError:
        raise ApiUsageError(
            f"AXW010: unknown dynamics {dynamics!r}; known: {sorted(DYNAMICS)}"
        ) from None


@dataclass
class Input:
    """Declares the input interface dimension (encoder output size)."""

    size: int


@dataclass
class Readout:
    """Declares the readout interface dimension (target classes/tokens)."""

    size: int


class _Dense:
    def __init__(self, in_size: int, out_size: int, key):
        self.weight = jax.random.normal(key, (in_size, out_size), dtype=jnp.float32) * jnp.sqrt(
            2.0 / in_size
        )
        self.bias = jnp.zeros((out_size,), dtype=jnp.float32)

    def __call__(self, x):
        return x @ self.weight + self.bias


class ConnectomeBlock:
    """Composable connectome computing block with time-stepped dynamics.

    Runs ``n_steps`` dynamics steps per forward pass over the sparse
    substrate. With ``dynamics='rate'`` and ``n_steps=1`` this is
    equivalent to the plain sparse :class:`ConnectomeLayer`.
    """

    def __init__(
        self,
        brain,
        dynamics: str | DynamicsModel = "rate",
        trainable_edges: bool = False,
        n_steps: int = 1,
        seed: int | None = None,
        selection: NeuronSelection | None = None,
    ):
        if dynamics not in ("rate",) and n_steps != 1 and trainable_edges:
            import warnings

            warnings.warn(
                "AXW007: multi-step spiking dynamics with trainable_edges is not "
                "surrogate-gradient enabled yet; gradients through the spiking "
                "path will be zero. Use dynamics='rate' or n_steps=1 for "
                "trainable edge learning.",
                stacklevel=2,
            )
        self.brain = brain
        self.dynamics = resolve_dynamics(dynamics)
        self.layer = ConnectomeLayer(
            brain.graph, trainable_edges=trainable_edges, selection=selection
        )
        self.selection = selection
        self.n_active = self.layer.n_neurons
        self.n_steps = n_steps
        self.seed = seed
        self._state = None

    def __call__(self, x):
        xa = jnp.asarray(x)
        if xa.shape[-1] != self.n_active:
            raise ApiUsageError(
                f"AXW010: expected last dimension {self.n_active}, got {xa.shape[-1]}"
            )
        np_x = np.asarray(xa)
        # The layer caches the (possibly selection-restricted) CSR; reuse it
        # instead of recomputing selection.weights() on every call.
        W = self.layer.graph_weights
        n = self.n_active
        batch_shape = np_x.shape[:-1]
        if self._state is None:
            self._state = self.dynamics.initial_state(n, batch_shape)
        activity = np_x
        for _ in range(self.n_steps):
            activity, self._state = self.dynamics.step(self._state, np_x, W)
        y = jnp.asarray(activity, dtype=xa.dtype)
        y = y * self.layer.gain
        return y


class BrainModel:
    """High-level supervised facade: Input -> connectome dynamics -> Readout.

    Training modes:
      - Mode 1 (frozen): ``trainable_edges=False`` — only interface params train.
      - Mode 2 (synaptic): ``trainable_edges=True`` — W = W0 + dW, topology fixed.
    """

    def __init__(
        self,
        brain,
        dynamics: str | DynamicsModel = "rate",
        trainable_edges: bool = False,
        train_dynamics: bool = False,
        learning: str | None = None,
        seed: int | None = None,
        selection: NeuronSelection | None = None,
    ):
        self.brain = brain
        self.selection = selection
        self.block = ConnectomeBlock(
            brain,
            dynamics=dynamics,
            trainable_edges=trainable_edges,
            seed=seed,
            selection=selection,
        )
        self._io_size = self.block.n_active
        self.input_proj: _Dense | None = None
        self.readout: _Dense | None = None
        self.train_dynamics = train_dynamics
        self.learning = learning
        self.seed = seed
        self._key = jax.random.PRNGKey(0 if seed is None else seed)

    @property
    def n_active(self) -> int:
        """Neurons the model computes over (sub-network size or full brain)."""
        return self._io_size

    def connect(self, module):
        """Register an Input or Readout interface module."""
        if isinstance(module, Input):
            self._key, subkey = jax.random.split(self._key)
            self.input_proj = _Dense(module.size, self._io_size, subkey)
        elif isinstance(module, Readout):
            self._key, subkey = jax.random.split(self._key)
            self.readout = _Dense(self._io_size, module.size, subkey)
        else:
            raise ApiUsageError("AXW010: connect() accepts Input or Readout instances")
        return self

    def __call__(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)
        y = self.block(x)
        if self.readout is not None:
            y = self.readout(y)
        return y
