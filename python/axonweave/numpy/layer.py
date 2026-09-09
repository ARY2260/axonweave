import numpy as np

from ..core.selection import NeuronSelection
from ..errors import ApiUsageError


class ConnectomeLayer:
    def __init__(self, graph, trainable_edges=False, gain=1.0, selection=None):
        if selection is not None:
            if not isinstance(selection, NeuronSelection):
                raise ApiUsageError(
                    "AXW010: selection must be a NeuronSelection from brain.graph.neurons")
            self.graph_weights = selection.weights()
            self.selection_body_ids = selection.body_ids.copy()
        else:
            self.graph_weights = graph.weights
            self.selection_body_ids = None
        self.trainable_edges = trainable_edges
        self.gain = gain
        self.edge_weight = self.graph_weights.data.copy() if trainable_edges else None

    def __call__(self, x):
        x = np.asarray(x)
        n = self.graph_weights.shape[0]
        if x.shape[-1] != n:
            from ..errors import ApiUsageError
            raise ApiUsageError(
                f"AXW010: expected last dimension {n}, got {x.shape[-1]}")
        m = self.graph_weights
        if self.edge_weight is not None:
            m = m.copy()
            m.data = self.edge_weight
        # Support ND input by folding leading axes into the feature matrix.
        lead_shape = x.shape[:-1]
        n = x.shape[-1]
        flat = x.reshape(-1, n)
        y = (flat @ m) * self.gain
        return y.reshape(*lead_shape, n)
