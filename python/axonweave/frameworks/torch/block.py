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
            import warnings
            warnings.warn(
                "AXW007: ConnectomeBlock's reference path runs dynamics on numpy "
                "arrays and cannot carry torch gradients across steps. For "
                "gradient-based training through spiking dynamics use "
                "BrainModel (which routes through the differentiable torch "
                "path) with SurrogateLIF/SurrogateAdaptiveLIF dynamics, or "
                "keep n_steps=1 with dynamics='rate' here.",
                stacklevel=2,
            )
        self.brain = brain
        self.dynamics = resolve_dynamics(dynamics)
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
    """High-level model: Encoder -> stateful connectome runtime -> Readout.

    Preferred construction is explicit and declarative — the encoder,
    dynamics, and readout are passed in and the wiring is done for you:

        model = BrainModel(
            brain=brain,
            encoder=VectorEncoder(8, 256),
            dynamics=LIF(),
            readout=ClassificationHead(256, 4),
        )

    The previous interface (``connect(Input(...))`` / ``connect(Readout(...))``)
    keeps working; constructor injection takes precedence when both are used.

    Temporal semantics (backend-neutral):

        model.reset_state()
        for x_t in stream:
            y_t = model.step(x_t)          # state persists between steps

        seq = model.forward_sequence(x)    # == sequential step() calls
        state = model.get_state()          # branch / replay
        model.step(x); model.set_state(state)
        model.detach_state()               # truncated BPTT boundary

    Exposes ``model.encoder``, ``model.brain``, ``model.dynamics``,
    ``model.readout`` and ``model.runtime``.

    Training modes:
      - Mode 1 (frozen): ``trainable_edges=False`` — only interface params train.
      - Mode 2 (synaptic): ``trainable_edges=True`` — W = W0 + dW, topology fixed.
      - Mode 3 (neuron params): ``train_dynamics=True`` — gain/bias per neuron train.
      - Mode 5 (hybrid): combine with ``learning`` plasticity rule.
    """

    def __init__(self, brain, dynamics: str | DynamicsModel = "rate",
                 trainable_edges: bool = False, train_dynamics: bool = False,
                 learning: str | None = None, seed: int | None = None,
                 selection: NeuronSelection | None = None,
                 encoder=None, readout=None):
        super().__init__()
        self.brain = brain
        self.selection = selection
        resolved = resolve_dynamics(dynamics)
        self.block = ConnectomeBlock(brain, dynamics=resolved,
                                     trainable_edges=trainable_edges, seed=seed,
                                     selection=selection)
        # Stateful runtime view over the same substrate/dynamics. For
        # surrogate-spiking dynamics this is the differentiable torch cell
        # (BPTT-capable); otherwise it is the reference-semantics bridge.
        from .bptt import TorchSurrogateLIF
        from .runtime_bridge import TorchStatefulRuntime
        from ...dynamics import SurrogateAdaptiveLIF, SurrogateLIF

        self._differentiable = isinstance(
            resolved, (SurrogateLIF, SurrogateAdaptiveLIF))
        if self._differentiable:
            self.runtime = TorchSurrogateLIF(
                resolved, trainable_edges=trainable_edges,
            ).attach_graph(self.block.layer.graph_weights)
        else:
            self.runtime = TorchStatefulRuntime(brain.graph, resolved)
        # Interface projections target the block's neuron space (sub-network
        # when a selection is given, full brain otherwise).
        self._io_size = self.block.n_active
        self.input_proj: nn.Linear | None = None
        self.readout: nn.Linear | None = None
        self.train_dynamics = train_dynamics
        self.learning = learning
        self.seed = seed
        if encoder is not None:
            self.connect(encoder)
        if readout is not None:
            self.connect(readout)
        self._encoder_module = encoder
        self._readout_module = readout

    @property
    def n_active(self) -> int:
        """Neurons the model computes over (sub-network size or full brain)."""
        return self._io_size

    @property
    def encoder(self):
        """The declared encoder (None when wired via connect(Input))."""
        return self._encoder_module

    @property
    def dynamics(self):
        """The active dynamics model."""
        return self.block.dynamics

    def connect(self, module):
        """Register an Input/Readout interface or a protocol encoder/readout."""
        from .interfaces import Input, Readout

        if isinstance(module, Input):
            self.input_proj = nn.Linear(module.size, self._io_size)
        elif isinstance(module, Readout):
            self.readout = nn.Linear(self._io_size, module.size)
        elif hasattr(module, "input_shape") and hasattr(module, "output_size"):
            # Encoder-protocol object: project its declared output_size onto
            # the connectome neuron space.
            if len(module.input_shape) != 1:
                raise ApiUsageError(
                    f"AXW010: encoder input_shape must be 1-D for BrainModel; "
                    f"got {module.input_shape}. Use forward_sequence for "
                    "time-series encoders.")
            self.input_proj = nn.Linear(module.output_size, self._io_size)
            self._encoder_module = module
        elif hasattr(module, "n_source") or hasattr(module, "n_outputs") or hasattr(module, "n_classes"):
            n_out = getattr(module, "n_classes", None) or getattr(
                module, "n_outputs", None) or getattr(module, "vocab_size", None)
            if n_out is None:
                raise ApiUsageError("AXW010: readout module lacks a declared output size")
            self.readout = nn.Linear(self._io_size, n_out)
            self._readout_module = module
        else:
            raise ApiUsageError(
                "AXW010: connect() accepts Input, Readout, or encoder/readout "
                "protocol objects")
        return self

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)
        y = self.block(x)
        if self.readout is not None:
            y = self.readout(y)
        return y

    # -- temporal semantics ---------------------------------------------------

    def reset_state(self) -> "BrainModel":
        """Clear carried state (episode/session boundary)."""
        self.runtime.reset_state()
        self.block._state = None
        return self

    def step(self, x_t):
        """One timestep (encoder -> runtime -> readout); state persists."""
        if self.input_proj is not None:
            x_t = self.input_proj(x_t)
        y = self.runtime.step(x_t)
        if self.readout is not None:
            y = self.readout(y)
        return y

    def forward_sequence(self, x):
        """Run ``[..., T, F]``; returns ``[..., T, output]``.

        Equivalent to sequential :meth:`step` calls from a reset state under
        deterministic execution (enforced by tests on the reference path).
        """
        if self.input_proj is not None:
            x = self.input_proj(x)
        y = self.runtime.forward_sequence(x)
        if self.readout is not None:
            y = self.readout(y)
        return y

    def get_state(self):
        """Deep-copied runtime state (branchable, replayable)."""
        return self.runtime.get_state()

    def set_state(self, state) -> "BrainModel":
        """Restore a previously captured state."""
        self.runtime.set_state(state)
        return self

    def detach_state(self) -> "BrainModel":
        """Detach carried gradients (truncated BPTT boundary)."""
        self.runtime.detach_state()
        return self

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
