"""Single dispatch layer for the Rust core (``axonweave._native``).

Every computationally meaningful primitive in the library routes through this
module when the compiled extension is available. When it is not, identical
NumPy/SciPy fallbacks (``_HAS_NATIVE = False``) keep the package fully
functional so the pure-python path and the compiled path produce the same
public results.

Both execution paths are first-class: the ``_numpy_*`` functions implement the
reference semantics unconditionally (also used by :mod:`tests.test_native_runtime`
to prove native/NumPy equivalence in CI), and the public dispatchers pick the
compiled kernel when ``_HAS_NATIVE`` is true.
"""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse

try:
    from . import _native as _NATIVE

    _HAS_NATIVE = True
except ImportError:
    _NATIVE = None
    _HAS_NATIVE = False


# --------------------------------------------------------------------------
# Graph: CSR construction, matmuls, submatrix, fingerprint (reference impls)
# --------------------------------------------------------------------------
def _numpy_sparse_matmul(csr: sparse.csr_matrix, x: np.ndarray) -> np.ndarray:
    return csr.dot(x).astype(np.float32)


def _numpy_sparse_matmul_transpose(csr: sparse.csr_matrix, x: np.ndarray) -> np.ndarray:
    out = csr.T.dot(x)
    return np.asarray(out, dtype=np.float32).ravel()


def _numpy_csr_matmul_2d(csr: sparse.csr_matrix, x: np.ndarray) -> np.ndarray:
    return (x @ csr.T).astype(np.float32)


def _numpy_csr_matmul_2d_transpose(csr: sparse.csr_matrix, x: np.ndarray) -> np.ndarray:
    return (x @ csr).astype(np.float32)


def _numpy_build_csr(rows, cols, weights, n_rows, n_cols):
    coo = sparse.coo_matrix(
        (np.asarray(weights, dtype=np.float32), (rows, cols)), shape=(n_rows, n_cols)
    )
    return coo.tocsr().astype(np.float32)


def _numpy_csr_submatrix(
    csr: sparse.csr_matrix, row_sel: np.ndarray, col_sel: np.ndarray
) -> sparse.csr_matrix:
    m = csr[row_sel][:, col_sel].tocsr()
    m.sort_indices()
    return m.astype(np.float32)


def _numpy_csr_fingerprint(csr: sparse.csr_matrix, body_ids: np.ndarray) -> str:
    m = csr.tocsr()
    m.sum_duplicates()
    m = m.sorted_indices()
    h = hashlib.sha256()
    h.update(np.asarray(m.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(body_ids, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(m.indptr, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(m.indices, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(m.data, dtype=np.float32).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Dynamics (reference impls)
# --------------------------------------------------------------------------
def _numpy_lif_step(v, refrac, current, t, tau, v_rest, v_threshold, v_reset, refractory, dt):
    t_new = t + dt
    can_spike = refrac <= t_new
    dv = (-(v - v_rest) + current) * (dt / tau)
    v_step = np.where(can_spike, v + dv, v)
    spiked = can_spike & (v_step >= v_threshold)
    v_out = np.where(spiked, v_reset, v_step)
    refrac_out = np.where(spiked, t_new + refractory, refrac)
    return spiked.astype(np.float32), v_out.astype(np.float32), refrac_out.astype(np.float32)


def _numpy_adaptive_lif_step(
    v, refrac, threshold, current, t, tau, v_rest, v_threshold, v_reset,
    refractory, tau_adapt, delta_threshold, dt,
):
    t_new = t + dt
    can_spike = refrac <= t_new
    dv = (-(v - v_rest) + current) * (dt / tau)
    v_step = np.where(can_spike, v + dv, v)
    spiked = can_spike & (v_step >= threshold)
    v_out = np.where(spiked, v_reset, v_step)
    th_relaxed = threshold + (v_threshold - threshold) * (dt / tau_adapt)
    th_out = np.where(spiked, th_relaxed + delta_threshold, th_relaxed)
    refrac_out = np.where(spiked, t_new + refractory, refrac)
    return (
        spiked.astype(np.float32),
        v_out.astype(np.float32),
        th_out.astype(np.float32),
        refrac_out.astype(np.float32),
    )


def _numpy_rate_step(current, gain, baseline):
    return (baseline + gain * current).astype(np.float32)


def _numpy_surrogate_forward(v, threshold):
    return np.where(v >= threshold, np.float32(1.0), np.float32(0.0))


def _numpy_surrogate_backward(v, threshold, kind, k, width):
    x = v - threshold
    if kind == 0:  # sigmoid
        xc = np.clip(x * k, -20.0, 20.0)
        sigma = 1.0 / (1.0 + np.exp(-xc))
        return (k * sigma * (1.0 - sigma)).astype(np.float32)
    if kind == 1:  # atan
        return (k / (1.0 + (np.pi * k * x) ** 2)).astype(np.float32)
    if kind == 2:  # piecewise
        return np.where(np.abs(x) <= 1.0 / k, np.float32(k), np.float32(0.0))
    if kind == 3:  # ste
        return np.where(np.abs(x) <= width, np.float32(1.0), np.float32(0.0))
    raise ValueError(f"surrogate kind must be 0..3, got {kind}")


def _numpy_surrogate_lif_step(
    v, refrac, current, t, tau, v_rest, v_threshold, v_reset, refractory, dt,
    kind, k, width,
):
    t_new = t + dt
    can_spike = refrac <= t_new
    dv = (-(v - v_rest) + current) * (dt / tau)
    v_step = np.where(can_spike, v + dv, v)
    gradient = _numpy_surrogate_backward(v_step, v_threshold, kind, k, width)
    gradient = np.where(can_spike, gradient, np.float32(0.0))
    spiked = can_spike & (v_step >= v_threshold)
    v_out = np.where(spiked, v_reset, v_step)
    refrac_out = np.where(spiked, t_new + refractory, refrac)
    return spiked.astype(np.float32), v_out.astype(np.float32), refrac_out.astype(np.float32), gradient


def _numpy_surrogate_adaptive_lif_step(
    v, refrac, threshold, current, t, tau, v_rest, v_threshold, v_reset,
    refractory, tau_adapt, delta_threshold, dt, kind, k, width,
):
    t_new = t + dt
    can_spike = refrac <= t_new
    dv = (-(v - v_rest) + current) * (dt / tau)
    v_step = np.where(can_spike, v + dv, v)
    gradient = _numpy_surrogate_backward(v_step, threshold, kind, k, width)
    gradient = np.where(can_spike, gradient, np.float32(0.0))
    spiked = can_spike & (v_step >= threshold)
    v_out = np.where(spiked, v_reset, v_step)
    th_relaxed = threshold + (v_threshold - threshold) * (dt / tau_adapt)
    th_out = np.where(spiked, th_relaxed + delta_threshold, th_relaxed)
    refrac_out = np.where(spiked, t_new + refractory, refrac)
    return (
        spiked.astype(np.float32),
        v_out.astype(np.float32),
        th_out.astype(np.float32),
        refrac_out.astype(np.float32),
        gradient,
    )


def _numpy_stdp_update(
    csr, pre_trace, post_trace, pre_activity, post_activity, a_plus, a_minus,
    tau_pre, tau_post, dt, reward=None, w_min=None, w_max=None,
):
    decay_pre = np.exp(np.float32(-dt) / np.float32(tau_pre))
    decay_post = np.exp(np.float32(-dt) / np.float32(tau_post))
    pre_prev = pre_trace.copy()
    pre_new = pre_trace * decay_pre + pre_activity
    post_new = post_trace * decay_post + post_activity
    if csr is not None and csr.nnz:
        rows = np.repeat(np.arange(csr.shape[0]), np.diff(csr.indptr))
        cols = csr.indices
        ltp = a_plus * np.outer(post_activity, pre_prev)
        ltd = a_minus * np.outer(post_new, pre_activity)
        d = (ltp - ltd)[rows, cols]
        mult = 1.0 if reward is None else reward
        data = csr.data.astype(np.float32) + mult * d.astype(np.float32)
        if w_min is not None:
            data = np.maximum(data, w_min)
        if w_max is not None:
            data = np.minimum(data, w_max)
    else:
        data = csr.data.astype(np.float32)
    return data.astype(np.float32), pre_new.astype(np.float32), post_new.astype(np.float32)


def _numpy_vesicle_release_step(
    vp, conc, pre, rel, dec, sto, sign, w,
):
    will_release = sto < (rel * pre)
    release_amount = vp * will_release.astype(np.float64)
    conc_out = np.clip(conc * dec + release_amount, 0.0, 1.0)
    vp_out = np.clip(vp - release_amount, 0.0, 1.0)
    vp_out = vp_out + (1.0 - vp_out) * (1.0 - dec)
    per = conc_out * sign * w
    return conc_out, vp_out, per, per.sum()


def _numpy_nt_currents(sign, weights, pre_activity):
    return sign * weights * pre_activity


def _numpy_receptor_step(
    g, pre, voltage, kind, decay_time_constant, reverse_potential, gain, sign,
    mg_concentration, mg_slope, mg_offset, dt,
):
    if kind == 0 or kind == 1:
        current = g * reverse_potential
    elif kind == 2:
        mg_block = 1.0 / (1.0 + mg_concentration * np.exp(-mg_slope * (voltage - mg_offset)))
        current = g * mg_block * reverse_potential
    elif kind == 3:
        current = g * gain * sign
    else:
        raise ValueError(f"receptor kind must be 0..3, got {kind}")
    g_new = g + (pre - g) * (dt / decay_time_constant)
    return current.astype(np.float32), g_new.astype(np.float32)


def _numpy_delay_ticks(d, dt, max_delay_ms, n_slots):
    return np.maximum(np.round(d / dt).astype(np.int64), 0).clip(max=n_slots - 1)


class NumpyDelayBuffer:
    """NumPy ring buffer mirroring the compiled ``DelayRing`` interface."""

    def __init__(self, n_slots: int, n_neurons: int):
        self._n_slots = max(1, int(n_slots))
        self._n_neurons = int(n_neurons)
        self.buffers = np.zeros((self._n_slots, self._n_neurons), dtype=np.float32)
        self._write_pos = np.zeros(self._n_neurons, dtype=np.int64)

    @property
    def write_pos(self):
        return self._write_pos

    def n_slots(self):
        return self._n_slots

    def n_neurons(self):
        return self._n_neurons

    def reset(self):
        self.buffers.fill(0.0)
        self._write_pos.fill(0)

    def step(self, pre, pre_idx, post_idx, delay_ticks, weights):
        slot = int(self._write_pos[0]) % self._n_slots
        self.buffers[slot] = np.asarray(pre, dtype=np.float32)
        self._write_pos += 1
        return self.read(pre_idx, post_idx, delay_ticks, weights)

    def read(self, pre_idx, post_idx, delay_ticks, weights):
        out = np.zeros(self._n_neurons, dtype=np.float32)
        read_tick = self._write_pos[pre_idx] - delay_ticks - 1
        valid = read_tick >= 0
        if np.any(valid):
            v_pre = pre_idx[valid]
            v_post = post_idx[valid]
            v_slot = read_tick[valid] % self._n_slots
            v_w = np.asarray(weights, dtype=np.float32)[valid]
            pre_vals = self.buffers[v_slot, v_pre]
            np.add.at(out, v_post, pre_vals * v_w)
        return out


def create_delay_ring(n_slots, n_neurons):
    if _HAS_NATIVE:
        return _NATIVE.DelayRing(int(n_slots), int(n_neurons))
    return NumpyDelayBuffer(n_slots, n_neurons)


def _numpy_apply_delayed_propagation(
    coo_row, coo_col, coo_data, pre_activity, delay_ticks, n_slots, n_rows, n_cols,
):
    ring = np.zeros((n_slots, n_rows), dtype=np.float32)
    ring[0] = pre_activity
    read_tick = 1 - delay_ticks - 1
    valid = read_tick >= 0
    out = np.zeros(n_cols, dtype=np.float32)
    if np.any(valid):
        v_pre = coo_row[valid].astype(np.int64)
        v_post = coo_col[valid].astype(np.int64)
        v_slot = read_tick[valid].astype(np.int64) % n_slots
        v_w = coo_data[valid].astype(np.float32)
        pre_vals = ring[v_slot, v_pre]
        np.add.at(out, v_post, pre_vals * v_w)
    return out


def _numpy_dense_matmul(a, b, m, k, n):
    return (a.reshape(m, k) @ b.reshape(k, n)).astype(np.float32)


def _numpy_row_absmax_normalize(x, m, k, eps):
    max_ = np.abs(x.reshape(m, k)).max(axis=-1, keepdims=True)
    return (x.reshape(m, k) / np.maximum(max_, eps)).astype(np.float32)


def _numpy_embed_lookup(ids, embedding, vocab, dim):
    return embedding[ids].astype(np.float32)


def _numpy_decode_argmax(vals, batch, n_in, n_actions):
    return np.argmax(vals.reshape(batch, n_in)[:, : min(n_actions, n_in)], axis=-1).astype(np.int64)


def _numpy_decode_clip(vals, batch, n_in, n_actions, low, high):
    return np.clip(vals.reshape(batch, n_in)[:, : min(n_actions, n_in)], low, high).astype(np.float32)


def _numpy_readout_logits(activity, weight, bias, batch, n_source, n_out):
    return (activity.reshape(batch, n_source) @ weight.reshape(n_source, n_out) + bias).astype(np.float32)


# --------------------------------------------------------------------------
# Public dispatchers
# --------------------------------------------------------------------------
def sparse_matmul(csr: sparse.csr_matrix, x: np.ndarray) -> np.ndarray:
    """W @ x (1-D, per-row accumulation of column-activity)."""
    x = np.asarray(x, dtype=np.float32)
    if _HAS_NATIVE:
        return _NATIVE.sparse_matmul(
            csr.data.astype(np.float32),
            csr.indices.astype(np.int64),
            csr.indptr.astype(np.int64),
            x,
            csr.shape[0],
            csr.shape[1],
        )
    return _numpy_sparse_matmul(csr, x)


def sparse_matmul_transpose(csr: sparse.csr_matrix, x: np.ndarray) -> np.ndarray:
    """x @ W = W.T @ x (1-D; row-activity propagates to columns)."""
    x = np.asarray(x, dtype=np.float32)
    if _HAS_NATIVE:
        return _NATIVE.sparse_matmul_transpose(
            csr.data.astype(np.float32),
            csr.indices.astype(np.int64),
            csr.indptr.astype(np.int64),
            x,
            csr.shape[0],
            csr.shape[1],
        )
    return _numpy_sparse_matmul_transpose(csr, x)


def csr_matmul_2d(csr: sparse.csr_matrix, x: np.ndarray) -> np.ndarray:
    """Batched W @ x^T -> (batch, n_rows). ``x`` has shape (batch, n_cols)."""
    x = np.asarray(x, dtype=np.float32)
    batch, _ = x.shape
    if _HAS_NATIVE:
        return _NATIVE.csr_matmul_2d(
            csr.data.astype(np.float32),
            csr.indices.astype(np.int64),
            csr.indptr.astype(np.int64),
            x,
            csr.shape[0],
            csr.shape[1],
        ).reshape(batch, csr.shape[0])
    return _numpy_csr_matmul_2d(csr, x)


def csr_matmul_2d_transpose(csr: sparse.csr_matrix, x: np.ndarray) -> np.ndarray:
    """Batched x @ W (dynamics path). ``x`` has shape (batch, n_rows)."""
    x = np.asarray(x, dtype=np.float32)
    batch, _ = x.shape
    if _HAS_NATIVE:
        return _NATIVE.csr_matmul_2d_transpose(
            csr.data.astype(np.float32),
            csr.indices.astype(np.int64),
            csr.indptr.astype(np.int64),
            x,
            csr.shape[0],
            csr.shape[1],
        ).reshape(batch, csr.shape[1])
    return _numpy_csr_matmul_2d_transpose(csr, x)


def build_csr_from_coo(
    rows: np.ndarray, cols: np.ndarray, weights: np.ndarray, shape: tuple[int, int]
) -> sparse.csr_matrix:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    n_rows, n_cols = shape

    if _HAS_NATIVE:
        data, indices, indptr = _NATIVE.build_csr(rows, cols, weights, n_rows, n_cols)
        return sparse.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))
    return _numpy_build_csr(rows, cols, weights, n_rows, n_cols)


def csr_submatrix(
    csr: sparse.csr_matrix, row_sel: np.ndarray, col_sel: np.ndarray
) -> sparse.csr_matrix:
    row_sel = np.asarray(row_sel, dtype=np.int64)
    col_sel = np.asarray(col_sel, dtype=np.int64)
    if _HAS_NATIVE:
        data, indices, indptr = _NATIVE.csr_submatrix(
            csr.data.astype(np.float32),
            csr.indices.astype(np.int64),
            csr.indptr.astype(np.int64),
            row_sel,
            col_sel,
            csr.shape[0],
            csr.shape[1],
        )
        return sparse.csr_matrix(
            (data, indices, indptr), shape=(len(row_sel), len(col_sel))
        )
    return _numpy_csr_submatrix(csr, row_sel, col_sel)


def csr_fingerprint(csr: sparse.csr_matrix, body_ids: np.ndarray) -> str:
    """Stable SHA-256 of shape, body ids, and canonical CSR payload."""
    m = csr.tocsr()
    m.sum_duplicates()
    m = m.sorted_indices()
    body_ids = np.ascontiguousarray(body_ids, dtype=np.int64)
    if _HAS_NATIVE:
        return _NATIVE.csr_fingerprint(
            np.ascontiguousarray(m.data, dtype=np.float32),
            np.ascontiguousarray(m.indices, dtype=np.int64),
            np.ascontiguousarray(m.indptr, dtype=np.int64),
            body_ids,
            m.shape[0],
            m.shape[1],
        )
    return _numpy_csr_fingerprint(csr, body_ids)


# --------------------------------------------------------------------------
# Dynamics
# --------------------------------------------------------------------------
def lif_step(
    v, refrac_until, current, t, tau, v_rest, v_threshold, v_reset, refractory, dt
):
    v = np.asarray(v, dtype=np.float32).ravel()
    refrac = np.asarray(refrac_until, dtype=np.float32).ravel()
    current = np.asarray(current, dtype=np.float32).ravel()
    if _HAS_NATIVE:
        return _NATIVE.lif_step(
            v, refrac, current, float(t), float(tau), float(v_rest), float(v_threshold),
            float(v_reset), float(refractory), float(dt),
        )
    return _numpy_lif_step(
        v, refrac, current, t, tau, v_rest, v_threshold, v_reset, refractory, dt
    )


def adaptive_lif_step(
    v, refrac_until, threshold, current, t, tau, v_rest, v_threshold, v_reset,
    refractory, tau_adapt, delta_threshold, dt,
):
    v = np.asarray(v, dtype=np.float32).ravel()
    refrac = np.asarray(refrac_until, dtype=np.float32).ravel()
    threshold = np.asarray(threshold, dtype=np.float32).ravel()
    current = np.asarray(current, dtype=np.float32).ravel()
    if _HAS_NATIVE:
        return _NATIVE.adaptive_lif_step(
            v, refrac, threshold, current, float(t), float(tau), float(v_rest),
            float(v_threshold), float(v_reset), float(refractory), float(tau_adapt),
            float(delta_threshold), float(dt),
        )
    return _numpy_adaptive_lif_step(
        v, refrac, threshold, current, t, tau, v_rest, v_threshold, v_reset,
        refractory, tau_adapt, delta_threshold, dt,
    )


def rate_step(current, gain, baseline):
    current = np.asarray(current, dtype=np.float32).ravel()
    if _HAS_NATIVE:
        return _NATIVE.rate_step(current, float(gain), float(baseline))
    return _numpy_rate_step(current, gain, baseline)


# --------------------------------------------------------------------------
# Surrogate gradients
# --------------------------------------------------------------------------
def surrogate_forward(v, threshold):
    v = np.asarray(v, dtype=np.float32).ravel()
    if _HAS_NATIVE:
        return _NATIVE.surrogate_forward(v, float(threshold))
    return _numpy_surrogate_forward(v, threshold)


def surrogate_backward(v, threshold, kind, k, width):
    v = np.asarray(v, dtype=np.float32).ravel()
    kind = int(kind)
    if kind not in (0, 1, 2, 3):
        raise ValueError(f"surrogate kind must be 0..3, got {kind}")
    if _HAS_NATIVE:
        return _NATIVE.surrogate_backward(v, float(threshold), kind, float(k), float(width))
    return _numpy_surrogate_backward(v, threshold, kind, k, width)


def surrogate_lif_step(
    v, refrac_until, current, t, tau, v_rest, v_threshold, v_reset, refractory, dt,
    kind, k, width,
):
    v = np.asarray(v, dtype=np.float32).ravel()
    refrac = np.asarray(refrac_until, dtype=np.float32).ravel()
    current = np.asarray(current, dtype=np.float32).ravel()
    kind = int(kind)
    if kind not in (0, 1, 2, 3):
        raise ValueError(f"surrogate kind must be 0..3, got {kind}")
    if _HAS_NATIVE:
        return _NATIVE.surrogate_lif_step(
            v, refrac, current, float(t), float(tau), float(v_rest), float(v_threshold),
            float(v_reset), float(refractory), float(dt), kind, float(k), float(width),
        )
    return _numpy_surrogate_lif_step(
        v, refrac, current, t, tau, v_rest, v_threshold, v_reset, refractory, dt,
        kind, k, width,
    )


def surrogate_adaptive_lif_step(
    v, refrac_until, threshold, current, t, tau, v_rest, v_threshold, v_reset,
    refractory, tau_adapt, delta_threshold, dt, kind, k, width,
):
    v = np.asarray(v, dtype=np.float32).ravel()
    refrac = np.asarray(refrac_until, dtype=np.float32).ravel()
    threshold = np.asarray(threshold, dtype=np.float32).ravel()
    current = np.asarray(current, dtype=np.float32).ravel()
    kind = int(kind)
    if kind not in (0, 1, 2, 3):
        raise ValueError(f"surrogate kind must be 0..3, got {kind}")
    if _HAS_NATIVE:
        return _NATIVE.surrogate_adaptive_lif_step(
            v, refrac, threshold, current, float(t), float(tau), float(v_rest),
            float(v_threshold), float(v_reset), float(refractory), float(tau_adapt),
            float(delta_threshold), float(dt), kind, float(k), float(width),
        )
    return _numpy_surrogate_adaptive_lif_step(
        v, refrac, threshold, current, t, tau, v_rest, v_threshold, v_reset,
        refractory, tau_adapt, delta_threshold, dt, kind, k, width,
    )


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------
def stdp_update(
    csr, pre_trace, post_trace, pre_activity, post_activity, a_plus, a_minus,
    tau_pre, tau_post, dt, reward=None, w_min=None, w_max=None,
):
    pre_trace = np.asarray(pre_trace, dtype=np.float32).ravel()
    post_trace = np.asarray(post_trace, dtype=np.float32).ravel()
    pre_activity = np.asarray(pre_activity, dtype=np.float32).ravel()
    post_activity = np.asarray(post_activity, dtype=np.float32).ravel()

    if _HAS_NATIVE:
        return _NATIVE.stdp_update(
            csr.data.astype(np.float32),
            csr.indices.astype(np.int64),
            csr.indptr.astype(np.int64),
            pre_trace,
            post_trace,
            pre_activity,
            post_activity,
            float(a_plus),
            float(a_minus),
            float(tau_pre),
            float(tau_post),
            float(dt),
            None if reward is None else float(reward),
            None if w_min is None else float(w_min),
            None if w_max is None else float(w_max),
        )
    return _numpy_stdp_update(
        csr, pre_trace, post_trace, pre_activity, post_activity, a_plus, a_minus,
        tau_pre, tau_post, dt, reward, w_min, w_max,
    )


# --------------------------------------------------------------------------
# Synapses / receptors
# --------------------------------------------------------------------------
def vesicle_release_step(
    vesicle_pool, concentration, pre_activity, release_probs, decay_rates,
    stochastic, signs, weights,
):
    vp = np.asarray(vesicle_pool, dtype=np.float64).ravel()
    conc = np.asarray(concentration, dtype=np.float64).ravel()
    pre = np.asarray(pre_activity, dtype=np.float64).ravel()
    rel = np.asarray(release_probs, dtype=np.float64).ravel()
    dec = np.asarray(decay_rates, dtype=np.float64).ravel()
    sto = np.asarray(stochastic, dtype=np.float64).ravel()
    sign = np.asarray(signs, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()

    if _HAS_NATIVE:
        return _NATIVE.vesicle_release_step(vp, conc, pre, rel, dec, sto, sign, w)
    return _numpy_vesicle_release_step(vp, conc, pre, rel, dec, sto, sign, w)


def nt_currents(signs, weights, pre_activity):
    sign = np.asarray(signs, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    pre_activity = np.asarray(pre_activity, dtype=np.float64).ravel()
    if _HAS_NATIVE:
        return _NATIVE.nt_currents(sign, weights, pre_activity)
    return _numpy_nt_currents(sign, weights, pre_activity)


def receptor_step(
    g, pre_activity, kind, decay_time_constant, reverse_potential, gain, sign,
    mg_concentration, mg_slope, mg_offset, voltage=None, dt=1.0,
):
    g = np.asarray(g, dtype=np.float32).ravel()
    pre = np.asarray(pre_activity, dtype=np.float32).ravel()
    if voltage is None:
        voltage = np.zeros_like(g)
    else:
        voltage = np.asarray(voltage, dtype=np.float32)
        if voltage.ndim == 0:
            voltage = np.full_like(g, float(voltage))
        voltage = voltage.ravel()
    kind = int(kind)
    if kind not in (0, 1, 2, 3):
        raise ValueError(f"receptor kind must be 0..3, got {kind}")
    if _HAS_NATIVE:
        return _NATIVE.receptor_step(
            g, pre, voltage, float(dt), kind, float(decay_time_constant),
            float(reverse_potential), float(gain), float(sign),
            float(mg_concentration), float(mg_slope), float(mg_offset),
        )
    return _numpy_receptor_step(
        g, pre, voltage, kind, decay_time_constant, reverse_potential, gain, sign,
        mg_concentration, mg_slope, mg_offset, dt,
    )


# --------------------------------------------------------------------------
# Delays
# --------------------------------------------------------------------------
def delay_ticks_from_ms(delays_ms, dt, max_delay_ms):
    d = np.asarray(delays_ms, dtype=np.float32)
    n_slots = max(1, int(np.ceil(max_delay_ms / dt)) + 1)
    if _HAS_NATIVE:
        return _NATIVE.delay_ticks_from_ms(d, float(dt), float(max_delay_ms))
    return _numpy_delay_ticks(d, dt, max_delay_ms, n_slots)


def apply_delayed_propagation(
    W: sparse.csr_matrix, pre_activity: np.ndarray, delays: np.ndarray,
    max_delay: float, dt: float,
) -> np.ndarray:
    """Stateless single-shot delayed propagation (compat with ``apply_delays``)."""
    pre_activity = np.asarray(pre_activity, dtype=np.float32)
    delays = np.asarray(delays, dtype=np.float32)

    if _HAS_NATIVE:
        return _NATIVE.apply_delays(
            W.data.astype(np.float32),
            W.indices.astype(np.int64),
            W.indptr.astype(np.int64),
            pre_activity,
            delays,
            float(max_delay),
            float(dt),
            W.shape[0],
            W.shape[1],
        )

    n_slots = int(np.ceil(max_delay / dt)) + 1
    n_rows, n_cols = W.shape
    coo = W.tocoo()
    delay_ticks = _numpy_delay_ticks(delays, dt, max_delay, n_slots).astype(np.int64)
    return _numpy_apply_delayed_propagation(
        coo.row, coo.col, coo.data, pre_activity, delay_ticks, n_slots, n_rows, n_cols,
    )


# --------------------------------------------------------------------------
# Dense encoding / decoding / readout
# --------------------------------------------------------------------------
def dense_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    m, k = a.shape
    _, n = b.shape
    if _HAS_NATIVE:
        return _NATIVE.dense_matmul(a.ravel(), b.ravel(), int(m), int(k), int(n)).reshape(m, n)
    return _numpy_dense_matmul(a.ravel(), b.ravel(), m, k, n)


def row_absmax_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    m, k = x.shape
    if _HAS_NATIVE:
        return _NATIVE.row_absmax_normalize(x.ravel(), int(m), int(k), float(eps)).reshape(m, k)
    return _numpy_row_absmax_normalize(x.ravel(), m, k, eps)


def embed_lookup(ids: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64).ravel()
    embedding = np.asarray(embedding, dtype=np.float32)
    vocab, dim = embedding.shape
    if _HAS_NATIVE:
        return _NATIVE.embed_lookup(ids, embedding.ravel(), int(vocab), int(dim)).reshape(
            ids.shape[0], dim
        )
    return _numpy_embed_lookup(ids, embedding, vocab, dim)


def decode_argmax(vals: np.ndarray, n_actions: int) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float32)
    batch, n_in = vals.shape
    if _HAS_NATIVE:
        return _NATIVE.decode_argmax(vals.ravel(), int(batch), int(n_in), int(n_actions))
    return _numpy_decode_argmax(vals.ravel(), batch, n_in, n_actions)


def decode_clip(vals: np.ndarray, n_actions: int, low: float, high: float) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float32)
    batch, n_in = vals.shape
    if _HAS_NATIVE:
        return _NATIVE.decode_clip(
            vals.ravel(), int(batch), int(n_in), int(n_actions), float(low), float(high)
        ).reshape(batch, min(n_actions, n_in))
    return _numpy_decode_clip(vals.ravel(), batch, n_in, n_actions, low, high)


def readout_logits(activity: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    activity = np.asarray(activity, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    batch, n_source = activity.shape
    n_out = weight.shape[1]
    if _HAS_NATIVE:
        return _NATIVE.readout_logits(
            activity.ravel(), weight.ravel(), bias.ravel(),
            int(batch), int(n_source), int(n_out),
        ).reshape(batch, n_out)
    return _numpy_readout_logits(activity.ravel(), weight.ravel(), bias.ravel(), batch, n_source, n_out)


# --------------------------------------------------------------------------
# Provisioning / file hashing
# --------------------------------------------------------------------------
def sha256_file(path) -> str:
    if _HAS_NATIVE:
        return _NATIVE.sha256_file(str(Path(path)))
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def md5_base64_of_file(path) -> str:
    if _HAS_NATIVE:
        return _NATIVE.md5_base64_file(str(Path(path)))
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode("ascii")


__all__ = [
    "_HAS_NATIVE",
    "NumpyDelayBuffer",
    "apply_delayed_propagation",
    "build_csr_from_coo",
    "create_delay_ring",
    "csr_fingerprint",
    "csr_matmul_2d",
    "csr_matmul_2d_transpose",
    "csr_submatrix",
    "decode_argmax",
    "decode_clip",
    "delay_ticks_from_ms",
    "dense_matmul",
    "embed_lookup",
    "adaptive_lif_step",
    "lif_step",
    "md5_base64_of_file",
    "nt_currents",
    "rate_step",
    "readout_logits",
    "receptor_step",
    "row_absmax_normalize",
    "sha256_file",
    "sparse_matmul",
    "sparse_matmul_transpose",
    "stdp_update",
    "surrogate_adaptive_lif_step",
    "surrogate_backward",
    "surrogate_forward",
    "surrogate_lif_step",
    "vesicle_release_step",
]