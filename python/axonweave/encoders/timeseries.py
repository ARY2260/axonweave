"""Vector and time-series encoders (Temporal Runtime Alpha).

Part of the stable encoder protocol: every encoder declares its
``input_shape`` (excluding batch), ``output_size`` and ``dtype`` so models
can validate wiring before execution.
"""
from __future__ import annotations

import numpy as np

from .. import native as _native
from ..errors import ApiUsageError


class VectorEncoder:
    """Fixed random projection from a feature vector to input currents.

    Declares ``input_shape=(input_dim,)``, ``output_size=output_dim`` and
    ``dtype='float32'`` per the encoder protocol. The projection is seeded
    and deterministic; it is an engineering assumption, not a biological one.
    """

    def __init__(self, input_dim: int, output_dim: int, seed: int = 0,
                 gain: float = 1.0):
        if input_dim <= 0 or output_dim <= 0:
            raise ApiUsageError(
                f"AXW010: encoder sizes must be positive; got input_dim={input_dim}, "
                f"output_dim={output_dim}")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.output_size = self.output_dim
        self.gain = float(gain)
        self.input_shape = (self.input_dim,)
        self.dtype = "float32"
        rng = np.random.default_rng(seed)
        # Seeded random projection, row-absmax normalized for stable currents.
        proj = rng.standard_normal((self.input_dim, self.output_dim))
        max_ = np.abs(proj).max(axis=0, keepdims=True)
        self.projection = (proj / np.maximum(max_, 1e-8)).astype(np.float32)

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != self.input_dim:
            raise ApiUsageError(
                f"AXW010: expected last dimension {self.input_dim}, got {x.shape[-1]}")
        out = _native.dense_matmul(x.reshape(-1, self.input_dim), self.projection)
        return (out * self.gain).reshape(*x.shape[:-1], self.output_dim)


class TimeSeriesEncoder:
    """Per-timestep encoder for ``[B, T, features]`` streams.

    Applies a fixed seeded projection at every timestep and exposes a
    configurable window: ``flatten_window > 1`` lets the current at step
    ``t`` carry the last ``k`` observations (delay-line style), otherwise
    the mapping is memoryless and recurrence lives in the connectome.

    Declares ``input_shape=(window, input_dim)``, ``output_size=output_dim``.
    """

    def __init__(self, input_dim: int, output_dim: int, window: int = 1,
                 seed: int = 0, gain: float = 1.0):
        if input_dim <= 0 or output_dim <= 0 or window <= 0:
            raise ApiUsageError(
                f"AXW010: encoder sizes must be positive; got input_dim={input_dim}, "
                f"output_dim={output_dim}, window={window}")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.output_size = self.output_dim
        self.window = int(window)
        self.gain = float(gain)
        self.input_shape = (self.window, self.input_dim)
        self.dtype = "float32"
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((self.window * self.input_dim, self.output_dim))
        max_ = np.abs(proj).max(axis=0, keepdims=True)
        self.projection = (proj / np.maximum(max_, 1e-8)).astype(np.float32)

    def _window(self, seq: np.ndarray) -> np.ndarray:
        """Build sliding windows [..., T, window*input_dim] from [..., T, F]."""
        T = seq.shape[-2]
        if T < self.window:
            raise ApiUsageError(
                f"AXW010: sequence length {T} is shorter than window={self.window}")
        k = self.window
        if k == 1:
            return seq
        shape = list(seq.shape)
        shape[-2] = T
        shape[-1] = k * self.input_dim
        out = np.zeros(shape, dtype=np.float32)
        for t in range(T):
            # Warm-up region: pad with zeros at the left edge of the window.
            seg = seq[..., max(0, t - k + 1):t + 1, :]
            width = seg.shape[-2] * self.input_dim
            out[..., t, k * self.input_dim - width:] = seg.reshape(
                *seq.shape[:-2], width)
        return out

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim < 2 or x.shape[-1] != self.input_dim:
            raise ApiUsageError(
                f"AXW010: TimeSeriesEncoder expects [..., T, {self.input_dim}], "
                f"got shape {x.shape}")
        w = self._window(x)
        out = _native.dense_matmul(w.reshape(-1, self.window * self.input_dim),
                                   self.projection)
        return (out * self.gain).reshape(*x.shape[:-2], x.shape[-2], self.output_dim)
