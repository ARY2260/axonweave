"""Tests for surrogate gradient estimators (framework-agnostic numpy).

Covers the four estimators and every guaranteed shape/value invariant of
the base :class:`SurrogateGradient` contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from axonweave.dynamics.surrogate import (
    ATanSurrogate,
    PiecewiseSurrogate,
    SigmoidSurrogate,
    StraightThroughEstimator,
    SurrogateGradient,
)

ESTIMATORS = [
    StraightThroughEstimator(),
    SigmoidSurrogate(),
    SigmoidSurrogate(k=20.0),
    ATanSurrogate(),
    ATanSurrogate(k=1.0),
    PiecewiseSurrogate(),
    PiecewiseSurrogate(k=2.0),
]


def test_base_forward_is_heaviside():
    v = np.array([-60.0, -50.0, -40.0], dtype=np.float32)
    out = SurrogateGradient().forward(v, -50.0)
    np.testing.assert_array_equal(out, np.array([0.0, 1.0, 1.0], dtype=np.float32))


def test_base_backward_not_implemented():
    with pytest.raises(NotImplementedError):
        SurrogateGradient().backward(np.zeros(3), 1.0)


@pytest.mark.parametrize("surrogate", ESTIMATORS)
def test_forward_is_binary_spike_indicator(surrogate):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(4, 5)).astype(np.float32)
    threshold = 0.0
    out = surrogate.forward(v, threshold)
    assert out.shape == v.shape
    assert set(np.unique(out)) <= {0.0, 1.0}
    np.testing.assert_array_equal(out, (v >= threshold).astype(np.float32))


@pytest.mark.parametrize("surrogate", ESTIMATORS)
def test_backward_shape_and_dtype(surrogate):
    rng = np.random.default_rng(1)
    v = rng.normal(size=(3, 6)).astype(np.float32)
    out = surrogate.backward(v, -1.0)
    assert out.shape == v.shape
    assert out.dtype == np.float32


def test_backward_scalar_broadcasts():
    out = SigmoidSurrogate().backward(np.float32(-1.0), 0.0)
    assert out.ndim == 0
    assert out.item() > 0.0


def test_ste_forward_heaviside_backward_unit_near_threshold():
    ste = StraightThroughEstimator(width=0.5)
    v = np.array([-1.0, -0.4, 0.0, 0.4, 1.0], dtype=np.float32)
    grad = ste.backward(v, 0.0)
    np.testing.assert_array_equal(grad, np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32))


def test_sigmoid_backward_matches_analytic_formula():
    k = 10.0
    sig = SigmoidSurrogate(k=k)
    x = np.array([-0.3, 0.0, 0.2], dtype=np.float32)
    sigma = 1.0 / (1.0 + np.exp(-k * x))
    expected = k * sigma * (1.0 - sigma)
    np.testing.assert_allclose(sig.backward(x, 0.0), expected, rtol=1e-6)


def test_sigmoid_backward_peak_at_threshold():
    grad = SigmoidSurrogate(k=5.0).backward(np.float32(0.0), 0.0)
    assert np.isclose(grad, 1.25, atol=1e-6)


def test_atan_backward_matches_analytic_formula():
    k = 3.0
    atan = ATanSurrogate(k=k)
    x = np.array([-0.4, 0.0, 0.5], dtype=np.float32)
    expected = k / (1.0 + (np.pi * k * x) ** 2)
    np.testing.assert_allclose(atan.backward(x, 0.0), expected, rtol=1e-6)


def test_atan_backward_at_threshold_equals_k():
    grad = ATanSurrogate(k=2.0).backward(np.float32(0.0), 0.0)
    assert np.isclose(grad, 2.0, atol=1e-6)


def test_piecewise_backward_is_zero_outside_window():
    graded = PiecewiseSurrogate(k=2.0)
    v = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    out = graded.backward(v, 0.0)
    np.testing.assert_array_equal(out, np.array([0.0, 2.0, 2.0, 2.0, 0.0], dtype=np.float32))
    assert np.isclose(graded.backward(np.float32(0.0), 0.0), 2.0)


def test_ste_backward_window_parameter():
    wide = StraightThroughEstimator(width=2.0)
    v = np.array([-1.9, -2.1], dtype=np.float32)
    np.testing.assert_array_equal(wide.backward(v, 0.0), np.array([1.0, 0.0], dtype=np.float32))


def test_sigmoid_backward_saturates_far_from_threshold():
    sig = SigmoidSurrogate(k=1.0)
    assert sig.backward(np.float32(-100.0), 0.0) < 1e-6
    assert sig.backward(np.float32(100.0), 0.0) < 1e-6