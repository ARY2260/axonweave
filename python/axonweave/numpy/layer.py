import numpy as np


class ConnectomeLayer:
    def __init__(self, graph, trainable_edges=False, gain=1.0):
        self.graph, self.trainable_edges, self.gain = graph, trainable_edges, gain
        self.edge_weight = graph.weights.data.copy() if trainable_edges else None

    def __call__(self, x):
        x = np.asarray(x)
        m = self.graph.weights
        if self.edge_weight is not None:
            m = m.copy()
            m.data = self.edge_weight
        # Support ND input by folding leading axes into the feature matrix.
        lead_shape = x.shape[:-1]
        n = x.shape[-1]
        flat = x.reshape(-1, n)
        y = (flat @ m) * self.gain
        return y.reshape(*lead_shape, n)
