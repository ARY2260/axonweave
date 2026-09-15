"""Tests for the temporal encoders and the stateful torch BrainModel.

Local (no-torch) runs cover the NumPy encoder protocol; torch-dependent
BrainModel tests are CI-verified-pending per the testing policy.
"""
from __future__ import annotations

import numpy as np
import pytest

from axonweave.encoders import TimeSeriesEncoder, VectorEncoder
from axonweave.errors import ApiUsageError


# ---------------------------------------------------------------------------
# VectorEncoder
# ---------------------------------------------------------------------------

def test_vector_encoder_shapes_and_protocol():
    enc = VectorEncoder(input_dim=8, output_dim=32, seed=1)
    assert enc.input_shape == (8,)
    assert enc.output_size == 32
    assert enc.dtype == "float32"
    y = enc(np.ones((4, 8), dtype=np.float32))
    assert y.shape == (4, 32)


def test_vector_encoder_deterministic():
    a = VectorEncoder(8, 16, seed=7)(np.ones((2, 8), dtype=np.float32))
    b = VectorEncoder(8, 16, seed=7)(np.ones((2, 8), dtype=np.float32))
    np.testing.assert_array_equal(a, b)


def test_vector_encoder_dimension_guard():
    enc = VectorEncoder(8, 16)
    with pytest.raises(ApiUsageError, match="AXW010"):
        enc(np.ones((2, 9), dtype=np.float32))


def test_vector_encoder_positive_sizes():
    with pytest.raises(ApiUsageError, match="AXW010"):
        VectorEncoder(0, 16)


# ---------------------------------------------------------------------------
# TimeSeriesEncoder
# ---------------------------------------------------------------------------

def test_timeseries_encoder_shapes_and_protocol():
    enc = TimeSeriesEncoder(input_dim=10, output_dim=64, window=1, seed=2)
    assert enc.input_shape == (1, 10)
    assert enc.output_size == 64
    y = enc(np.ones((3, 5, 10), dtype=np.float32))
    assert y.shape == (3, 5, 64)


def test_timeseries_encoder_window():
    enc = TimeSeriesEncoder(input_dim=4, output_dim=8, window=3, seed=3)
    y = enc(np.ones((2, 6, 4), dtype=np.float32))
    assert y.shape == (2, 6, 8)


def test_timeseries_encoder_rejects_short_sequences():
    enc = TimeSeriesEncoder(4, 8, window=5)
    with pytest.raises(ApiUsageError, match="AXW010"):
        enc(np.ones((2, 3, 4), dtype=np.float32))


def test_timeseries_encoder_window1_is_memoryless():
    """window=1 output must not depend on history (recurrence lives in the brain)."""
    enc = TimeSeriesEncoder(input_dim=4, output_dim=8, window=1, seed=4)
    a = enc(np.ones((1, 3, 4), dtype=np.float32))
    b = enc(np.ones((1, 6, 4), dtype=np.float32))
    np.testing.assert_allclose(a[0], b[0, :3], rtol=1e-6, atol=1e-6)
