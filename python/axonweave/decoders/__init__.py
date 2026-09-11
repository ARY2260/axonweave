"""Decoders (Phase 2): map neural readout activity to actions or predictions."""
from __future__ import annotations

import numpy as np

from .. import native as _native


class ActionDecoder:
    """Maps readout activity to discrete or continuous motor actions.

    Discrete mode returns the arg-max action index. Continuous mode returns
    a bounded real vector.
    """

    def __init__(self, actions: int, continuous: bool = False,
                 low: float = -1.0, high: float = 1.0, seed: int = 0,
                 select: str | None = None):
        self.actions = actions
        self.continuous = continuous
        self.low, self.high = low, high
        self.select = select
        rng = np.random.default_rng(seed)
        self.projection = (rng.standard_normal((actions, actions)) *
                           np.eye(actions, dtype=np.float32)).astype(np.float32)

    def __call__(self, activity, n_source: int | None = None) -> np.ndarray:
        a = np.asarray(activity, dtype=np.float32)
        if a.shape[-1] < self.actions:
            from ..errors import ApiUsageError
            raise ApiUsageError(
                f"AXW010: decoder needs at least {self.actions} readout neurons, got {a.shape[-1]}"
            )
        lead = a.shape[:-1]
        flat = a.reshape(-1, a.shape[-1])
        if self.continuous:
            return _native.decode_clip(flat, self.actions, self.low, self.high).reshape(*lead, self.actions)
        return _native.decode_argmax(flat, self.actions).reshape(lead)


class TokenDecoder:
    """Linear projection from neural readout to vocabulary logits."""

    def __init__(self, vocab_size: int, seed: int = 0, select: str | None = None):
        self.vocab_size = vocab_size
        self.select = select
        # Projection is created lazily because n_source depends on the readout
        # selection; keep it stable per instance.
        self._w = None
        self._seed = seed

    def __call__(self, activity) -> np.ndarray:
        a = np.asarray(activity, dtype=np.float32)
        n_source = a.shape[-1]
        if self._w is None:
            rng = np.random.default_rng(self._seed)
            self._w = (rng.standard_normal((n_source, self.vocab_size)) *
                       (1.0 / np.sqrt(n_source))).astype(np.float32)
        lead = a.shape[:-1]
        out = _native.dense_matmul(a.reshape(-1, n_source), self._w)
        return out.reshape(*lead, self.vocab_size)


# Alias for the common supervised head.
ClassificationHead = TokenDecoder
