import numpy as np
import torch
from torch import nn
from ..errors import UnsupportedDeviceError


class ConnectomeLayer(nn.Module):
    def __init__(self, graph, trainable_edges=False, learnable_gain=False,
                 bias=False, signal_policy=None, device=None):
        super().__init__()
        self.n_neurons = graph.n_neurons
        coo = graph.weights.tocoo()
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

    def forward(self, x):
        if x.shape[-1] != self.n_neurons:
            raise ValueError(f"AXW010: expected last dimension {self.n_neurons}, got {x.shape[-1]}")
        w = torch.sparse_coo_tensor(self.edge_index, self.edge_weight,
                                    (self.n_neurons, self.n_neurons), device=x.device).coalesce()
        y = torch.sparse.mm(w.transpose(0, 1), x.transpose(-1, -2)).transpose(-1, -2)
        y = y * self.gain
        if self.bias is not None:
            y = y + self.bias
        return y
