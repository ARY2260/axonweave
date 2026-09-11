from __future__ import annotations

import numpy as np

from .. import native as _native


class SensorEncoder:
    """Generic vector-to-current mapping for telemetry / robot sensors."""

    def __init__(self, n_sensors: int, n_target: int, seed: int = 0,
                 select: str | None = None):
        self.n_sensors = n_sensors
        self.n_target = n_target
        self.select = select
        rng = np.random.default_rng(seed)
        self.projection = (rng.standard_normal((n_sensors, n_target)) *
                           (1.0 / np.sqrt(n_sensors))).astype(np.float32)

    def __call__(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if x.shape[-1] != self.n_sensors:
            from ..errors import ApiUsageError
            raise ApiUsageError(
                f"AXW010: expected last dimension {self.n_sensors}, got {x.shape[-1]}"
            )
        lead = x.shape[:-1]
        flat = x.reshape(-1, self.n_sensors)
        flat = _native.row_absmax_normalize(flat)
        out = _native.dense_matmul(flat, self.projection)
        return out.reshape(*lead, self.n_target)
