from __future__ import annotations

import numpy as np

from .. import native as _native


class ImageEncoder:
    """Maps 2D/3D visual observations to currents for a target neuron group.

    The observation is flattened and linearly projected (fixed random
    projection by default, or a user-supplied projection matrix) to
    ``n_target`` neurons.
    """

    def __init__(self, shape: tuple[int, ...], n_target: int,
                 seed: int = 0, projection=None, normalize: bool = True,
                 select: str | None = None):
        self.shape = tuple(shape)
        self.n_input = int(np.prod(self.shape))
        self.n_target = n_target
        self.normalize = normalize
        self.select = select
        if projection is not None:
            self.projection = np.asarray(projection, dtype=np.float32)
            assert self.projection.shape == (self.n_input, n_target)
        else:
            rng = np.random.default_rng(seed)
            self.projection = (rng.standard_normal((self.n_input, n_target)) *
                               (1.0 / np.sqrt(self.n_input))).astype(np.float32)

    def __call__(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        lead = x.shape[:-len(self.shape)]
        flat = x.reshape(-1, self.n_input)
        if self.normalize:
            flat = _native.row_absmax_normalize(flat)
        out = _native.dense_matmul(flat, self.projection)
        return out.reshape(*lead, self.n_target)
