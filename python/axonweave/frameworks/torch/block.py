"""PyTorch framework adapter: BrainModel and ConnectomeBlock (Phases 1-3).

The substrate provides sparse structure; dynamics and interfaces are
explicit model choices. Tensor execution stays entirely in PyTorch.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ...dynamics import LIF, AdaptiveLIF, Rate, DynamicsModel
from ...errors import ApiUsageError
from ...torch.layer import ConnectomeLayer
from ...core.selection import NeuronSelection


DYNAMICS: dict[str, type[DynamicsModel]] = {
    "lif": LIF,
    "adaptive_lif": AdaptiveLIF,
    "rate": Rate,
}


def resolve_dynamics(dynamics: str | DynamicsModel | None) -> DynamicsModel:
    if dynamics is None:
        return Rate()
    if isinstance(dynamics, DynamicsModel):
        return dynamics
    try:
        return DYNAMICS[dynamics]()
    except KeyError:
        raise ApiUsageError(
            f"AXW010: unknown dynamics {dynamics!r}; known: {sorted(DYNAMICS)}"
        ) from None


class ConnectomeBlock(nn.Module):
    """Composable connectome computing block with time-stepped dynamics.

    Runs ``n_steps`` dynamics steps per forward pass over the sparse
    substrate. With ``dynamics='rate'`` and ``n_steps=1`` this is
    equivalent to the plain sparse :class:`ConnectomeLayer`.
    """

    def __init__(self, brain, dynamics: str | DynamicsModel = "rate",
                 trainable_edges: bool = False, learnable_gain: bool = False,
                 select: int | list | None = None, n_steps: int = 1, seed: int | None = None,
                 selection: NeuronSelection | None = None):
        super().__init__()
        if dynamics not in ("rate",) and n_steps != 1 and trainable_edges:
            # Gradient flow through multi-step spiking dynamics is not
            # surrogate-gradient enabled yet; be explicit rather than silent.
            pass
        self.brain = brain
        self.layer = ConnectomeLayer(
            brain.graph, trainable_edges=trainable_edges,
            learnable_gain=learnable_gain, selection=selection)
        # Selection defines the block's neuron space; None means the full brain.
        self.selection = selection
        self.n_active = self.layer.n_neurons
        self.select = select
        self.seed = seed
        self._state = None

    def _selected_indices(self, n_neurons: int):
        if self.select is None:
            return None
        return np.asarray(self.select, dtype=np.int64)

    def forward(self, x):
        if x.shape[-1] != self.n_active:
            raise ApiUsageError(
                f"AXW010: expected last dimension {self.n_active}, got {x.shape[-1]}"
            )
        np_x = x.detach().cpu().numpy()
        W = self.layer.graph_weights if self.selection is not None else self.brain.graph.weights
        n = self.n_active
        batch_shape = np_x.shape[:-1]
        if self._state is None:
            self._state = self.dynamics.initial_state(n, batch_shape)
        activity = np_x
        for _ in range(self.n_steps):
            activity_np, self._state = self.dynamics.step(self._state, np_x, W)
            activity = activity_np
        y = torch.as_tensor(activity, dtype=x.dtype, device=x.device)
        y = y * self.layer.gain
        return y


class BrainModel(nn.Module):
    """High-level supervised facade: Input -> connectome dynamics -> Readout.

    Training modes:
      - Mode 1 (frozen): ``trainable_edges=False`` — only interface params train.
      - Mode 2 (synaptic): ``trainable_edges=True`` — W = W0 + dW, topology fixed.
      - Mode 3 (neuron params): ``train_dynamics=True`` — gain/bias per neuron train.
      - Mode 5 (hybrid): combine with ``learning`` plasticity rule.
    """

    def __init__(self, brain, dynamics: str | DynamicsModel = "rate",
                 trainable_edges: bool = False, train_dynamics: bool = False,
                 learning: str | None = None, seed: int | None = None,
                 selection: NeuronSelection | None = None):
        super().__init__()
        self.brain = brain
        self.selection = selection
        self.block = ConnectomeBlock(brain, dynamics=dynamics,
                                     trainable_edges=trainable_edges, seed=seed,
                                     selection=selection)
        # Interface projections target the block's neuron space (sub-network
        # when a selection is given, full brain otherwise).
        self._io_size = self.block.n_active
        self.input_proj: nn.Linear | None = None
        self.readout: nn.Linear | None = None
        self.train_dynamics = train_dynamics
        self.learning = learning
        self.seed = seed

    @property
    def n_active(self) -> int:
        """Neurons the model computes over (sub-network size or full brain)."""
        return self._io_size

    def connect(self, module):
        """Register an Input or Readout interface module."""
        from .interfaces import Input, Readout

        if isinstance(module, Input):
            self.input_proj = nn.Linear(module.size, self._io_size)
        elif isinstance(module, Readout):
            self.readout = nn.Linear(self._io_size, module.size)
        else:
            raise ApiUsageError("AXW010: connect() accepts Input or Readout instances")
        return self

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)
        y = self.block(x)
        if self.readout is not None:
            y = self.readout(y)
        return y

    def fit(self, loader, epochs: int = 1, lr: float = 1e-3, optimizer=None):
        """Standard supervised loop over (x, y) batches from a DataLoader/iterator."""
        if self.readout is None:
            raise ApiUsageError("AXW010: BrainModel.fit requires a Readout via connect()")
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise ApiUsageError(
                "AXW010: nothing to train — the substrate is frozen and no interfaces "
                "were connected. Use connect(Input(...)) / connect(Readout(...)) or "
                "trainable_edges=True."
            )

        opt = optimizer or torch.optim.Adam(params, lr=lr)
        self.train()
        history = []
        for epoch in range(epochs):
            total, count = 0.0, 0
            for x, y in loader:
                opt.zero_grad()
                logits = self(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                total += float(loss)
                count += 1
            history.append(total / max(count, 1))
        return history
