"""Keras framework adapter: BrainLayer and ConnectomeBlock (Phase 6).

Keras-native mirror of the torch ``BrainModel``: the substrate provides
sparse structure; dynamics and interfaces are explicit model choices.
Tensor execution, gradients and device placement stay entirely in
TensorFlow/Keras. Compatible with the Functional and Sequential APIs and
``model.fit()``.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ...dynamics import AdaptiveLIF, LIF, Rate, DynamicsModel
from ...errors import ApiUsageError
from ...keras.layer import ConnectomeLayer


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


class KerasConnectomeBlock(keras.layers.Layer):
    """Connectome computing block with time-stepped dynamics (Keras).

    Wraps the sparse :class:`~axonweave.keras.ConnectomeLayer` and applies
    ``n_steps`` dynamics steps per call over the (possibly selected)
    sub-network. With ``dynamics='rate'`` and ``n_steps=1`` this reduces to
    plain sparse propagation.
    """

    def __init__(self, brain, dynamics: str | DynamicsModel = "rate",
                 trainable_edges: bool = False, learnable_gain: bool = False,
                 n_steps: int = 1, seed: int | None = None,
                 selection=None, **kwargs):
        super().__init__(**kwargs)
        self.brain = brain
        self.selection = selection
        self.layer = ConnectomeLayer(
            brain.graph, trainable_edges=trainable_edges,
            learnable_gain=learnable_gain, selection=selection)
        self.n_active = self.layer.n_neurons
        self.dynamics = resolve_dynamics(dynamics)
        self.n_steps = n_steps
        self.seed = seed
        self._state = None

    def build(self, input_shape):
        if input_shape[-1] != self.n_active:
            raise ApiUsageError(
                f"AXW010: expected last dimension {self.n_active}, got {input_shape[-1]}")
        super().build(input_shape)

    def call(self, x):
        np_x = x.numpy() if isinstance(x, tf.Tensor) else np.asarray(x)
        W = self.layer.graph_weights if self.selection is not None else self.brain.graph.weights
        n = self.n_active
        batch_shape = np_x.shape[:-1]
        if self._state is None:
            self._state = self.dynamics.initial_state(n, batch_shape)
        activity = np_x
        for _ in range(self.n_steps):
            activity, self._state = self.dynamics.step(self._state, np_x, W)
        y = tf.convert_to_tensor(np.asarray(activity, dtype=np.float32))
        return y * self.layer.gain

    def reset_state(self):
        """Clear recurrent dynamics state (call between episodes)."""
        self._state = None


class BrainLayer(keras.layers.Layer):
    """High-level Keras facade: Input -> connectome dynamics -> Readout.

    Keras mirror of the torch ``BrainModel``. Use ``connect()`` to declare
    interface dimensions, then compose with the Functional or Sequential
    API and train with ``model.fit()``.

    Training modes:
      - Mode 1 (frozen): ``trainable_edges=False`` — only interface params train.
      - Mode 2 (synaptic): ``trainable_edges=True`` — W = W0 + dW, topology fixed.
    """

    def __init__(self, brain, dynamics: str | DynamicsModel = "rate",
                 trainable_edges: bool = False, seed: int | None = None,
                 selection=None, **kwargs):
        super().__init__(**kwargs)
        self.brain = brain
        self.selection = selection
        self.block = KerasConnectomeBlock(
            brain, dynamics=dynamics, trainable_edges=trainable_edges,
            seed=seed, selection=selection)
        # Interface projections target the block's neuron space (sub-network
        # when a selection is given, full brain otherwise).
        self._io_size = self.block.n_active
        self.input_proj: keras.layers.Dense | None = None
        self.readout: keras.layers.Dense | None = None
        self.seed = seed

    @property
    def n_active(self) -> int:
        """Neurons the model computes over (sub-network size or full brain)."""
        return self._io_size

    def connect(self, module) -> "BrainLayer":
        """Register an Input or Readout interface descriptor."""
        from ...frameworks.interfaces import Input, Readout

        if isinstance(module, Input):
            self.input_proj = keras.layers.Dense(self._io_size, name="input_proj")
        elif isinstance(module, Readout):
            self.readout = keras.layers.Dense(module.size, name="readout")
        else:
            raise ApiUsageError("AXW010: connect() accepts Input or Readout instances")
        return self

    def call(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)
        y = self.block(x)
        if self.readout is not None:
            y = self.readout(y)
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "dynamics": getattr(self.block.dynamics, "name", "rate"),
            "trainable_edges": self.block.layer.trainable_edges,
            "n_active": self._io_size,
        })
        return config
