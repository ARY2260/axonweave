import numpy as np
import torch
from torch import nn
from ..errors import ApiUsageError, UnsupportedDeviceError
from ..core.selection import NeuronSelection


class ConnectomeLayer(nn.Module):
    """Sparse biological connectome as a PyTorch module.

    Propagates batched neuron activity through the substrate's CSR topology:
    ``y = (x @ W^T) * gain + bias`` with W's sparsity pattern fixed to the
    connectome (trainable edges change values, never the topology).

    Args:
        graph: the ``ConnectomeGraph`` (or a ``NeuronSelection``-derived graph).
        trainable_edges: expose edge weights as an ``nn.Parameter`` (Mode 2
            synaptic learning; topology stays fixed).
        learnable_gain: make the global output gain trainable.
        bias: per-neuron additive bias (trainable).
        signal_policy: optional ``SignalPolicy``. The sparse layer computes
            structural propagation only; wiring a receptor/sign model here is
            not yet supported, so passing one raises ``AXW007`` instead of
            being silently ignored.
        device: device to place the layer on (AXW004 if the backend rejects it).
        selection: ``NeuronSelection`` restricting the layer to a sub-network.
    """

    def __init__(self, graph, trainable_edges=False, learnable_gain=False,
                 bias=False, signal_policy=None, device=None, selection=None):
        super().__init__()
        if signal_policy is not None:
            import warnings
            warnings.warn(
                "AXW007: ConnectomeLayer computes structural sparse propagation "
                "only; signal_policy is accepted for API symmetry but is not "
                "applied. Use the signals/receptors APIs for receptor and "
                "neurotransmitter models.",
                stacklevel=2,
            )
        if selection is not None:
            if not isinstance(selection, NeuronSelection):
                raise ApiUsageError(
                    "AXW010: selection must be a NeuronSelection from brain.graph.neurons")
            sub = selection.weights()
            self.selection_body_ids = selection.body_ids.copy()
        else:
            sub = graph.weights
            self.selection_body_ids = None
        self.n_neurons = sub.shape[0]
        self.graph_weights = sub
        coo = sub.tocoo()
        self.register_buffer("edge_index", torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long))
        values = torch.tensor(coo.data, dtype=torch.float32)
        if trainable_edges:
            self.edge_weight = nn.Parameter(values)
        else:
            self.register_buffer("edge_weight", values)
        self.gain = nn.Parameter(torch.ones(())) if learnable_gain else nn.Parameter(torch.ones(()), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(self.n_neurons)) if bias else None
        if device is not None:
            try:
                self.to(device)
            except Exception as e:
                raise UnsupportedDeviceError(f"AXW004: backend rejected device={device!r}: {e}") from e
        self.signal_policy = signal_policy

    @property
    def graph_weights(self):
        """The (possibly selection-restricted) CSR matrix backing this layer."""
        return self._graph_weights

    @graph_weights.setter
    def graph_weights(self, value):
        self._graph_weights = value

    def extra_repr(self) -> str:
        parts = [f"n_neurons={self.n_neurons}", f"n_edges={self.edge_weight.numel()}"]
        if self.edge_weight.requires_grad:
            parts.append("trainable_edges=True")
        if isinstance(self.gain, nn.Parameter) and self.gain.requires_grad:
            parts.append("learnable_gain=True")
        if self.bias is not None:
            parts.append("bias=True")
        if self.selection_body_ids is not None:
            parts.append(f"selection={len(self.selection_body_ids)} neurons")
        return ", ".join(parts)

    def _sparse_w(self, device, dtype):
        """Build the coalesced sparse weight matrix in x's dtype/device."""
        w = torch.sparse_coo_tensor(
            self.edge_index, self.edge_weight.to(device=device, dtype=dtype),
            (self.n_neurons, self.n_neurons), device=device,
        ).coalesce()
        return w

    def forward(self, x):
        if x.shape[-1] != self.n_neurons:
            raise ApiUsageError(f"AXW010: expected last dimension {self.n_neurons}, got {x.shape[-1]}")
        w = self._sparse_w(x.device, x.dtype)
        # W^T @ x^T == (x @ W)^T: row-activity propagates to columns.
        y = torch.sparse.mm(w.t(), x.transpose(-1, -2).to(dtype=w.dtype)).transpose(-1, -2)
        y = y * self.gain
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype)
