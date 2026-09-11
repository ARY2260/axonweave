from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO
except ImportError as e:
    from ..errors import BackendUnavailableError

    raise BackendUnavailableError(
        "AXW006: axonweave.jax requires JAX; install axonweave[jax]"
    ) from e

from ..core.selection import NeuronSelection
from ..errors import ApiUsageError, UnsupportedDeviceError


def _resolve_device(device):
    if device is None:
        return None
    if isinstance(device, jax.Device):
        return device
    if not isinstance(device, str):
        raise UnsupportedDeviceError(
            f"AXW004: device={device!r} must be a jax.Device or a device string"
        )
    try:
        return jax.devices(device)[0]
    except Exception as e:
        raise UnsupportedDeviceError(f"AXW004: backend rejected device={device!r}: {e}") from e


class ConnectomeLayer:
    def __init__(
        self,
        graph,
        trainable_edges=False,
        gain=1.0,
        bias=False,
        signal_policy=None,
        selection=None,
        device=None,
    ):
        if selection is not None:
            if not isinstance(selection, NeuronSelection):
                raise ApiUsageError(
                    "AXW010: selection must be a NeuronSelection from brain.graph.neurons"
                )
            sub = selection.weights()
            self.selection_body_ids = selection.body_ids.copy()
        else:
            sub = graph.weights
            self.selection_body_ids = None
        self.n_neurons = sub.shape[0]
        self.device = _resolve_device(device)
        w = BCOO.from_scipy_sparse(sub)
        w = w.sum_duplicates(remove_zeros=False)
        self._indices_sorted = w.indices_sorted
        self._unique_indices = w.unique_indices
        self.indices = jax.device_put(w.indices, self.device)
        self.edge_weight = jax.device_put(w.data.astype(jnp.float32), self.device)
        self.gain = float(gain)
        self.bias = (
            jax.device_put(jnp.zeros(self.n_neurons, dtype=jnp.float32), self.device)
            if bias
            else None
        )
        self.signal_policy = signal_policy
        self.trainable_edges = trainable_edges
        self.shape = (self.n_neurons, self.n_neurons)

    def __call__(self, x):
        xa = jnp.asarray(x)
        if xa.shape[-1] != self.n_neurons:
            raise ApiUsageError(
                f"AXW010: expected last dimension {self.n_neurons}, got {xa.shape[-1]}"
            )
        lead = xa.shape[:-1]
        xf = xa.reshape(-1, self.n_neurons)
        w = BCOO(
            (self.edge_weight, self.indices),
            shape=self.shape,
            indices_sorted=self._indices_sorted,
            unique_indices=self._unique_indices,
        )
        y = (w @ xf.T).T
        y = y * self.gain
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*lead, self.n_neurons)
