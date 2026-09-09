"""Readouts (Phase 6): task-facing interfaces over neural activity.

A readout maps a chosen subset of neurons (or the full brain) onto a task
output: classification logits, regression values or token logits. Readouts
are framework-neutral (NumPy) so the experiment/agent path can use them
without torch; the torch side composes them via ``BrainModel.connect``.

The unified namespace ``axonweave.readout`` re-exports the decoder classes
under their readout names, giving one documented home for the output side
of the pipeline:

    encoder -> brain -> readout
"""
from __future__ import annotations

import numpy as np

from ..errors import ApiUsageError


class ClassificationReadout:
    """Linear classification head over selected neuron activity.

    Parameters
    ----------
    n_source:
        Number of readout neurons feeding the head (e.g. ``len(sel)`` for a
        neuron selection, or ``brain.n_neurons`` for the full brain).
    n_classes:
        Number of output classes.
    trainable:
        When True the weight matrix is exposed as ``.weight`` for host
        framework optimizers to pick up (NumPy reference path updates it
        in-place via :meth:`update`).
    """

    def __init__(self, n_source: int, n_classes: int, trainable: bool = False,
                 seed: int = 0):
        if n_source <= 0 or n_classes <= 0:
            raise ApiUsageError(
                f"AXW010: readout sizes must be positive; got n_source={n_source}, "
                f"n_classes={n_classes}")
        self.n_source = n_source
        self.n_classes = n_classes
        self.trainable = trainable
        rng = np.random.default_rng(seed)
        self.weight = (rng.standard_normal((n_source, n_classes)) *
                       (1.0 / np.sqrt(n_source))).astype(np.float32)
        self.bias = np.zeros(n_classes, dtype=np.float32)

    def __call__(self, activity) -> np.ndarray:
        a = np.asarray(activity, dtype=np.float32)
        if a.shape[-1] != self.n_source:
            raise ApiUsageError(
                f"AXW010: expected last dimension {self.n_source}, got {a.shape[-1]}")
        return a @ self.weight + self.bias

    def update(self, grad_weight: np.ndarray, grad_bias: np.ndarray | None = None,
               lr: float = 1e-3) -> None:
        """Reference SGD step (NumPy path); torch/TF users use their optimizers."""
        if not self.trainable:
            raise ApiUsageError(
                "AXW010: readout is not trainable; construct with trainable=True")
        self.weight -= lr * np.asarray(grad_weight, dtype=np.float32)
        if grad_bias is not None:
            self.bias -= lr * np.asarray(grad_bias, dtype=np.float32)


class RegressionReadout:
    """Bounded linear readout for scalar/vector targets."""

    def __init__(self, n_source: int, n_outputs: int = 1, trainable: bool = False,
                 seed: int = 0):
        if n_source <= 0 or n_outputs <= 0:
            raise ApiUsageError(
                f"AXW010: readout sizes must be positive; got n_source={n_source}, "
                f"n_outputs={n_outputs}")
        self.n_source = n_source
        self.n_outputs = n_outputs
        self.trainable = trainable
        rng = np.random.default_rng(seed)
        self.weight = (rng.standard_normal((n_source, n_outputs)) *
                       (1.0 / np.sqrt(n_source))).astype(np.float32)
        self.bias = np.zeros(n_outputs, dtype=np.float32)

    def __call__(self, activity) -> np.ndarray:
        a = np.asarray(activity, dtype=np.float32)
        if a.shape[-1] != self.n_source:
            raise ApiUsageError(
                f"AXW010: expected last dimension {self.n_source}, got {a.shape[-1]}")
        return a @ self.weight + self.bias

    def update(self, grad_weight: np.ndarray, grad_bias: np.ndarray | None = None,
               lr: float = 1e-3) -> None:
        if not self.trainable:
            raise ApiUsageError(
                "AXW010: readout is not trainable; construct with trainable=True")
        self.weight -= lr * np.asarray(grad_weight, dtype=np.float32)
        if grad_bias is not None:
            self.bias -= lr * np.asarray(grad_bias, dtype=np.float32)


# Token readout keeps the decoder implementation; renamed for the task API.
from ..decoders import ActionDecoder, ClassificationHead, TokenDecoder  # noqa: E402,F401


class TokenReadout(TokenDecoder):
    """Next-token logits over neural readout activity.

    Same signature as :class:`~axonweave.decoders.TokenDecoder`:
    ``TokenReadout(n_source, vocab_size, seed=0)``.
    """

    def __init__(self, n_source: int, vocab_size: int, seed: int = 0):
        super().__init__(vocab_size=vocab_size, seed=seed)
        self.n_source = n_source

    def __call__(self, activity) -> np.ndarray:
        a = np.asarray(activity, dtype=np.float32)
        if a.shape[-1] != self.n_source:
            raise ApiUsageError(
                f"AXW010: expected last dimension {self.n_source}, got {a.shape[-1]}")
        return super().__call__(a)


class ActionReadout(ActionDecoder):
    """Motor-action readout over neural readout activity.

    Same signature as :class:`~axonweave.decoders.ActionDecoder`:
    ``ActionReadout(n_source, actions, continuous=False, seed=0)``.
    """

    def __init__(self, n_source: int, actions: int, continuous: bool = False,
                 low: float = -1.0, high: float = 1.0, seed: int = 0):
        if n_source < actions:
            raise ApiUsageError(
                f"AXW010: readout needs at least {actions} source neurons, got {n_source}")
        super().__init__(actions=actions, continuous=continuous, low=low,
                         high=high, seed=seed)
        self.n_source = n_source

    def __call__(self, activity) -> np.ndarray:
        a = np.asarray(activity, dtype=np.float32)
        if a.shape[-1] != self.n_source:
            raise ApiUsageError(
                f"AXW010: expected last dimension {self.n_source}, got {a.shape[-1]}")
        return super().__call__(a)


__all__ = [
    "ClassificationReadout", "RegressionReadout", "TokenReadout",
    "ActionReadout", "ClassificationHead", "TokenDecoder", "ActionDecoder",
]
