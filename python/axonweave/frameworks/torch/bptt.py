"""Gradient routing through the stateful torch path (Phase VI: BPTT/surrogate).

Pure-torch spiking cells that carry a differentiable state graph across
timesteps, using the surrogate-gradient contract from
:mod:`axonweave.dynamics.surrogate` (sigmoid / atan / piecewise / STE) to
route gradients through the binary spike nonlinearity.

Design:

- The membrane state is a torch tensor recreated each step as
  ``f(previous_state)`` so autograd builds a true recurrent graph across
  the sequence (BPTT) — no numpy round-trip on this path.
- The spike is ``SpikeSurrogate.apply(v, threshold, kind, k, width)``:
  hard threshold forward, surrogate derivative backward, with the surrogate
  formulas mirroring ``native.surrogate_backward`` exactly.
- Propagation uses the persistent torch sparse matrix built once by the
  caller (no per-step COO construction).
- ``detach_state()`` truncates the graph (truncated BPTT).
"""
from __future__ import annotations

import torch
from torch import nn

from ...dynamics import AdaptiveLIF, LIF


class SpikeSurrogate(torch.autograd.Function):
    """Hard spike forward, surrogate derivative backward.

    The backward formulas mirror ``native.surrogate_backward`` (kinds
    0=sigmoid, 1=atan, 2=piecewise, 3=STE) so the torch path and the
    NumPy/native references implement the same mathematics.
    """

    @staticmethod
    def forward(ctx, v, threshold, kind, k, width):
        ctx.save_for_backward(v, threshold)
        ctx.kind, ctx.k, ctx.width = int(kind), float(k), float(width)
        return (v >= threshold).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        v, threshold = ctx.saved_tensors
        x = v - threshold
        kind, k = ctx.kind, ctx.k
        if kind == 0:      # sigmoid
            xc = torch.clamp(x * k, -20.0, 20.0)
            s = torch.sigmoid(xc)
            g = k * s * (1.0 - s)
        elif kind == 1:    # atan
            g = k / (1.0 + (torch.pi * k * x) ** 2)
        elif kind == 2:    # piecewise
            g = torch.where(x.abs() <= 1.0 / k,
                            torch.full_like(x, k), torch.zeros_like(x))
        elif kind == 3:    # STE
            g = torch.where(x.abs() <= ctx.width,
                            torch.ones_like(x), torch.zeros_like(x))
        else:
            raise ValueError(f"surrogate kind must be 0..3, got {kind}")
        return grad_out * g, None, None, None, None


class _SparseProp(nn.Module):
    """Persistent sparse propagation layer: y = x @ W^T with fixed pattern."""

    def __init__(self, weights, trainable_edges: bool):
        super().__init__()
        coo = weights.tocoo()
        n = weights.shape[0]
        self.n_neurons = n
        self.register_buffer(
            "edge_index",
            torch.tensor(
                __import__("numpy").vstack([coo.row, coo.col]), dtype=torch.long,
            ),
        )
        values = torch.tensor(coo.data, dtype=torch.float32)
        self.edge_weight = (
            nn.Parameter(values) if trainable_edges
            else nn.Parameter(values, requires_grad=False)
        )

    def forward(self, x):
        w = torch.sparse_coo_tensor(
            self.edge_index, self.edge_weight.to(x.dtype),
            (self.n_neurons, self.n_neurons), device=x.device,
        ).coalesce()
        return torch.sparse.mm(w.t(), x.transpose(-1, -2)).transpose(-1, -2)


class TorchSurrogateLIF(nn.Module):
    """Differentiable recurrent LIF for BPTT (torch-native).

    Wraps the parameter set of :class:`axonweave.dynamics.LIF` (or its
    ``SurrogateLIF`` form, which carries the surrogate choice). Exposes the
    runtime contract: ``reset_state`` / ``step`` / ``forward_sequence`` /
    ``get_state`` / ``set_state`` / ``detach_state``.
    """

    def __init__(self, dynamics: LIF, trainable_edges: bool = False,
                 learnable_gain: bool = False):
        super().__init__()
        self.tau = float(dynamics.tau)
        self.v_rest = float(dynamics.v_rest)
        self.v_threshold = float(dynamics.v_threshold)
        self.v_reset = float(dynamics.v_reset)
        self.refractory = float(dynamics.refractory)
        self.dt = float(dynamics.dt)
        self.name = getattr(dynamics, "name", "lif")
        self.adaptive = isinstance(dynamics, AdaptiveLIF)
        if self.adaptive:
            self.tau_adapt = float(dynamics.tau_adapt)
            self.delta_threshold = float(dynamics.delta_threshold)
        # Surrogate: from SurrogateLIF config, else a sigmoid default.
        surr = getattr(dynamics, "surrogate", None)
        self.surrogate_kind = getattr(surr, "kind", 0)
        self.surrogate_k = getattr(surr, "k", 5.0)
        self.surrogate_width = getattr(surr, "width", 0.5)
        self._prop = None
        self.trainable_edges = trainable_edges
        self.learnable_gain = learnable_gain

    def attach_graph(self, weights):
        """Bind the substrate's CSR (once); builds persistent sparse buffers."""
        self._prop = _SparseProp(weights, self.trainable_edges)
        self.gain = (
            nn.Parameter(torch.ones(()))
            if self.learnable_gain
            else nn.Parameter(torch.ones(()), requires_grad=False)
        )
        return self

    # -- state ---------------------------------------------------------------

    def reset_state(self, batch_shape: tuple[int, ...] = ()) -> None:
        self._state = None
        self._batch_shape = batch_shape

    def _initial(self, batch_shape, device, dtype):
        shape = (*batch_shape, self._prop.n_neurons)
        state = {
            "v": torch.full(shape, self.v_rest, device=device, dtype=dtype),
            "refrac": torch.zeros(shape, device=device, dtype=dtype),
            "t": torch.zeros((), device=device, dtype=dtype),
        }
        if self.adaptive:
            state["threshold"] = torch.full(
                shape, self.v_threshold, device=device, dtype=dtype)
        return state

    def get_state(self):
        return {k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in (self._state or {}).items()}

    def set_state(self, state):
        self._state = {k: (v.clone() if torch.is_tensor(v) else v)
                       for k, v in state.items()}

    def detach_state(self):
        """Truncated-BPTT boundary: cut the recurrent autograd graph."""
        if self._state is not None:
            self._state = {k: (v.detach() if torch.is_tensor(v) else v)
                           for k, v in self._state.items()}
        return self

    # -- dynamics -------------------------------------------------------------

    def _cell(self, state, current):
        """One differentiable LIF cell; returns (spikes, new_state)."""
        t = state["t"] + self.dt
        can_spike = state["refrac"] <= t
        dv = (-(state["v"] - self.v_rest) + current) * (self.dt / self.tau)
        v_step = torch.where(can_spike, state["v"] + dv, state["v"])
        threshold = state.get("threshold", self.v_threshold)
        spikes = SpikeSurrogate.apply(
            v_step, threshold, self.surrogate_kind, self.surrogate_k,
            self.surrogate_width,
        )
        spikes = spikes * can_spike.to(spikes.dtype)
        v_out = torch.where(spikes > 0,
                            torch.full_like(v_step, self.v_reset), v_step)
        refrac_out = torch.where(spikes > 0,
                                 torch.full_like(state["refrac"], t + self.refractory),
                                 state["refrac"])
        new = {"v": v_out, "refrac": refrac_out, "t": t}
        if self.adaptive:
            th_relaxed = threshold + (self.v_threshold - threshold) * (
                self.dt / self.tau_adapt)
            th_out = torch.where(spikes > 0,
                                 th_relaxed + self.delta_threshold, th_relaxed)
            new["threshold"] = th_out
        return spikes, new

    def step(self, x_t):
        """One timestep; ``x_t`` is ``[..., F]`` input currents into the net."""
        if self._state is None:
            self._state = self._initial(
                tuple(x_t.shape[:-1]), x_t.device, x_t.dtype)
        current = self._prop(x_t)
        spikes, self._state = self._cell(self._state, current)
        return spikes * self.gain

    def forward_sequence(self, x):
        """``[..., T, F]`` -> ``[..., T, N]`` with a carried gradient graph."""
        lead = x.shape[:-2]
        T = x.shape[-2]
        xf = x.reshape(-1, T, x.shape[-1])
        outs = []
        self.reset_state(batch_shape=tuple(lead))
        for t in range(T):
            outs.append(self.step(xf[:, t, :]))
        return torch.stack(outs, dim=1).reshape(*lead, T, -1)
