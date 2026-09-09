import tensorflow as tf

from ..core.selection import NeuronSelection


class ConnectomeLayer(tf.keras.layers.Layer):
    def __init__(self, graph, trainable_edges=False, learnable_gain=False, use_bias=False,
                 selection=None, **kwargs):
        super().__init__(**kwargs)
        if selection is not None:
            if not isinstance(selection, NeuronSelection):
                raise ValueError(
                    "AXW010: selection must be a NeuronSelection from brain.graph.neurons")
            sub = selection.weights()
            self.selection_body_ids = selection.body_ids.copy()
        else:
            sub = graph.weights
            self.selection_body_ids = None
        c = sub.tocoo()
        self.n_neurons = sub.shape[0]
        self.indices = tf.constant(list(zip(c.row.tolist(), c.col.tolist())), tf.int64)
        self.initial = c.data.astype("float32")
        self.trainable_edges = trainable_edges
        self.learnable_gain = learnable_gain
        self.use_bias = use_bias

    def build(self, input_shape):
        if input_shape[-1] != self.n_neurons:
            raise ValueError(f"AXW010: expected last dimension {self.n_neurons}, got {input_shape[-1]}")
        self.edge_weight = self.add_weight(name="edge_weight", shape=(len(self.initial),),
            initializer=tf.keras.initializers.Constant(self.initial), trainable=self.trainable_edges)
        self.gain = self.add_weight(name="gain", shape=(), initializer="ones", trainable=self.learnable_gain)
        self.bias = self.add_weight(name="bias", shape=(self.n_neurons,), initializer="zeros", trainable=True) if self.use_bias else None

    def call(self, x):
        s = tf.sparse.reorder(tf.SparseTensor(self.indices, self.edge_weight, (self.n_neurons, self.n_neurons)))
        y = tf.transpose(tf.sparse.sparse_dense_matmul(s, tf.transpose(x))) * self.gain
        return y + self.bias if self.bias is not None else y
