import numpy as np
import tensorflow as tf

from ..core.selection import NeuronSelection
from ..errors import ApiUsageError


class ConnectomeLayer(tf.keras.layers.Layer):
    """Sparse biological connectome as a Keras layer.

    Propagates batched neuron activity through the substrate's CSR topology:
    ``y = (x @ W^T) * gain + bias`` with W's sparsity pattern fixed to the
    connectome (trainable edges change values, never the topology).

    Args:
        graph: the ``ConnectomeGraph`` (or a ``NeuronSelection``-derived graph).
        trainable_edges: expose edge weights as a trainable Keras variable
            (Mode 2 synaptic learning; topology stays fixed).
        learnable_gain: make the global output gain trainable.
        use_bias: per-neuron additive bias (trainable).
        signal_policy: optional ``SignalPolicy``. The sparse layer computes
            structural propagation only; wiring a receptor/sign model here is
            not yet supported, so passing one raises ``AXW007`` instead of
            being silently ignored.
        selection: ``NeuronSelection`` restricting the layer to a sub-network.
        **kwargs: forwarded to ``tf.keras.layers.Layer``.
    """

    def __init__(self, graph, trainable_edges=False, learnable_gain=False, use_bias=False,
                 signal_policy=None, selection=None, **kwargs):
        super().__init__(**kwargs)
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
        c = sub.tocoo()
        self.n_neurons = sub.shape[0]
        self.graph_weights = sub
        # Vectorized index construction (no Python list-of-tuples round trip).
        self.indices = tf.constant(
            np.stack([c.row, c.col], axis=1), tf.int64,
        )
        self.initial = c.data.astype("float32")
        self.trainable_edges = trainable_edges
        self.learnable_gain = learnable_gain
        self.use_bias = use_bias
        self.signal_policy = signal_policy

    def build(self, input_shape):
        if input_shape[-1] != self.n_neurons:
            raise ApiUsageError(f"AXW010: expected last dimension {self.n_neurons}, got {input_shape[-1]}")
        self.edge_weight = self.add_weight(name="edge_weight", shape=(len(self.initial),),
            initializer=tf.keras.initializers.Constant(self.initial), trainable=self.trainable_edges)
        self.gain = self.add_weight(name="gain", shape=(), initializer="ones", trainable=self.learnable_gain)
        self.bias = self.add_weight(name="bias", shape=(self.n_neurons,), initializer="zeros", trainable=True) if self.use_bias else None
        super().build(input_shape)

    def call(self, x):
        s = tf.sparse.reorder(
            tf.SparseTensor(self.indices, tf.cast(self.edge_weight, x.dtype), (self.n_neurons, self.n_neurons))
        )
        y = tf.transpose(tf.sparse.sparse_dense_matmul(s, tf.transpose(x))) * self.gain
        return y + self.bias if self.bias is not None else y

    def get_config(self):
        """Keras serialization: enough to clone the layer's structure.

        The sparse topology itself is not embedded in the config; a cloned
        layer re-resolves it from the same graph object, so deserialization
        requires the substrate to be loadable (fingerprint validation at
        load time still applies).
        """
        config = super().get_config()
        config.update({
            "trainable_edges": self.trainable_edges,
            "learnable_gain": self.learnable_gain,
            "use_bias": self.use_bias,
            "n_neurons": self.n_neurons,
        })
        return config
