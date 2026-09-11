from __future__ import annotations

import numpy as np

from .. import native as _native


class SurrogateGradient:
    """Base class for surrogate gradients of a binary spike function.

    ``forward`` returns the hard spike decision (1 when the membrane
    potential is at or above threshold, else 0). ``backward`` returns a
    differentiable approximation of that spike function's gradient and is
    designed to be framework-agnostic (numpy). Framework adapters wrap
    these methods in their autograd machinery.
    """

    name = "surrogate"
    kind = -1
    k = 1.0
    width = 0.5

    def forward(self, v, threshold):
        v = np.asarray(v, dtype=np.float32)
        return _native.surrogate_forward(v, float(threshold)).reshape(v.shape)

    def backward(self, v, threshold):
        raise NotImplementedError


class StraightThroughEstimator(SurrogateGradient):
    """Straight-through estimator: hard spike forward, unit gradient near threshold."""

    name = "ste"
    kind = 3

    def __init__(self, width: float = 0.5):
        super().__init__()
        self.width = width

    def backward(self, v, threshold):
        v = np.asarray(v, dtype=np.float32)
        return _native.surrogate_backward(
            v, float(threshold), self.kind, float(self.k), float(self.width)
        ).reshape(v.shape)


class SigmoidSurrogate(SurrogateGradient):
    """Sigmoid spike surrogate: ``sigma(x) = 1 / (1 + exp(-k * x))``."""

    name = "sigmoid"
    kind = 0

    def __init__(self, k: float = 5.0):
        super().__init__()
        self.k = k

    def backward(self, v, threshold):
        v = np.asarray(v, dtype=np.float32)
        return _native.surrogate_backward(
            v, float(threshold), self.kind, float(self.k), float(self.width)
        ).reshape(v.shape)


class ATanSurrogate(SurrogateGradient):
    """Arctangent spike surrogate: ``1/pi * atan(pi * k * x) + 0.5``."""

    name = "atan"
    kind = 1

    def __init__(self, k: float = 2.0):
        super().__init__()
        self.k = k

    def backward(self, v, threshold):
        v = np.asarray(v, dtype=np.float32)
        return _native.surrogate_backward(
            v, float(threshold), self.kind, float(self.k), float(self.width)
        ).reshape(v.shape)


class PiecewiseSurrogate(SurrogateGradient):
    """Linear spike surrogate in ``[-1/k, 1/k]`` and zero elsewhere."""

    name = "piecewise"
    kind = 2

    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def backward(self, v, threshold):
        v = np.asarray(v, dtype=np.float32)
        return _native.surrogate_backward(
            v, float(threshold), self.kind, float(self.k), float(self.width)
        ).reshape(v.shape)