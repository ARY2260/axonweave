"""PyTorch bridge for the temporal runtime semantics.

Adapts :class:`axonweave.runtime.ConnectomeRuntime` semantics to torch
tensors: explicit state capture/restore, ``detach_state`` for truncated
BPTT, and step/forward_sequence equivalence on the torch path.
"""
from __future__ import annotations

import torch
from torch import nn

from ...runtime import ConnectomeRuntime, RuntimeState


class TorchStatefulRuntime(nn.Module):
    """Stateful single-step runtime over the sparse substrate (torch).

    Delegates the numeric step to the NumPy reference runtime on detached
    arrays (identical semantics, CPU-executed), then maps the result back to
    the caller's device/dtype. Trainable parameters (edge weights, gain)
    stay torch ``nn.Parameter`` objects on the wrapping block; gradients
    through the *stateful recurrent* path require the surrogate/bptt work
    (Phase VI) and are not silently approximated here.
    """

    def __init__(self, graph, dynamics):
        super().__init__()
        self._ref = ConnectomeRuntime(graph, dynamics=dynamics)
        self.dynamics = dynamics

    @property
    def n_neurons(self) -> int:
        return self._ref.n_neurons

    @property
    def timestep(self) -> int:
        return self._ref.timestep

    def reset_state(self) -> None:
        self._ref.reset_state()

    def get_state(self) -> RuntimeState:
        return self._ref.get_state()

    def set_state(self, state: RuntimeState) -> None:
        self._ref.set_state(state)

    def detach_state(self) -> "TorchStatefulRuntime":
        # Reference path carries no autograd graph; explicit for API parity.
        return self

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """One timestep on a ``[..., F]`` tensor; state persists."""
        np_x = x_t.detach().cpu().numpy()
        y = self._ref.step(np_x)
        return torch.as_tensor(y, dtype=x_t.dtype, device=x_t.device)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., T, F]`` -> ``[..., T, N]``; step-equivalent, deterministic."""
        np_x = x.detach().cpu().numpy()
        y = self._ref.forward_sequence(np_x)
        return torch.as_tensor(y, dtype=x.dtype, device=x.device)
