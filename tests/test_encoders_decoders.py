"""Tests for encoders and decoders (Phase 2)."""
import numpy as np
import pytest

from axonweave.decoders import ActionDecoder, ClassificationHead, TokenDecoder
from axonweave.encoders import ImageEncoder, SensorEncoder, TokenEncoder


# ---------------------------------------------------------------------------
# ImageEncoder
# ---------------------------------------------------------------------------

def test_image_encoder_output_shape():
    enc = ImageEncoder(shape=(8, 8, 3), n_target=16, seed=1)
    out = enc(np.zeros((2, 8, 8, 3), dtype=np.float32))
    assert out.shape == (2, 16)
    assert out.dtype == np.float32


def test_image_encoder_deterministic_with_seed():
    a = ImageEncoder(shape=(4, 4, 3), n_target=8, seed=7)
    b = ImageEncoder(shape=(4, 4, 3), n_target=8, seed=7)
    x = np.ones((1, 4, 4, 3), dtype=np.float32)
    np.testing.assert_array_equal(a(x), b(x))


def test_image_encoder_normalizes():
    enc = ImageEncoder(shape=(4, 4, 1), n_target=4, normalize=True)
    small = enc(np.full((1, 4, 4, 1), 0.5, dtype=np.float32))
    big = enc(np.full((1, 4, 4, 1), 100.0, dtype=np.float32))
    np.testing.assert_allclose(small, big, rtol=1e-4)  # scale-invariant


def test_image_encoder_batched_3d_input():
    enc = ImageEncoder(shape=(8, 8, 3), n_target=16)
    out = enc(np.zeros((2, 3, 8, 8, 3), dtype=np.float32))
    assert out.shape == (2, 3, 16)


# ---------------------------------------------------------------------------
# TokenEncoder
# ---------------------------------------------------------------------------

def test_token_encoder_shape():
    enc = TokenEncoder(vocab_size=100, n_target=8)
    out = enc(np.array([1, 2, 3]))
    assert out.shape == (3, 8)


def test_token_encoder_batched_sequence():
    enc = TokenEncoder(vocab_size=100, n_target=8)
    out = enc(np.array([[1, 2], [3, 4]]))
    assert out.shape == (2, 2, 8)


def test_token_encoder_rejects_out_of_vocab():
    enc = TokenEncoder(vocab_size=10, n_target=4)
    with pytest.raises(ValueError, match="AXW010"):
        enc(np.array([10]))


# ---------------------------------------------------------------------------
# SensorEncoder
# ---------------------------------------------------------------------------

def test_sensor_encoder_shape_and_validation():
    enc = SensorEncoder(n_sensors=4, n_target=8)
    assert enc(np.zeros((2, 4), dtype=np.float32)).shape == (2, 8)
    with pytest.raises(ValueError, match="AXW010"):
        enc(np.zeros((2, 5), dtype=np.float32))


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

def test_action_decoder_discrete_argmax():
    dec = ActionDecoder(actions=4)
    activity = np.array([[0.1, 0.9, 0.2, 0.3]], dtype=np.float32)
    assert dec(activity).tolist() == [1]


def test_action_decoder_continuous_bounded():
    dec = ActionDecoder(actions=3, continuous=True, low=-1.0, high=1.0)
    out = dec(np.array([[10.0, -10.0, 0.5]], dtype=np.float32))
    assert out.shape == (1, 3)
    assert (out <= 1.0).all() and (out >= -1.0).all()


def test_action_decoder_needs_enough_neurons():
    dec = ActionDecoder(actions=5)
    with pytest.raises(ValueError, match="AXW010"):
        dec(np.zeros((1, 3), dtype=np.float32))


def test_token_decoder_logits_shape():
    dec = TokenDecoder(vocab_size=50)
    out = dec(np.zeros((2, 6), dtype=np.float32))
    assert out.shape == (2, 50)


def test_token_decoder_lazy_projection_is_stable():
    dec = TokenDecoder(vocab_size=7)
    a = np.random.default_rng(0).random((2, 6)).astype(np.float32)
    first = dec(a)
    np.testing.assert_array_equal(dec(a), first)  # same projection reused


def test_classification_head_alias():
    assert ClassificationHead is TokenDecoder
