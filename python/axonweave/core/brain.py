from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUBSTRATE_ID = "male-cns:v1.0"


def substrate_fingerprint(graph) -> str:
    """Stable identity hash of a loaded connectome (weights + body IDs).

    Deterministic for identical graph content regardless of matrix
    canonicalization order of duplicate entries.
    """
    from .. import native as _native

    return _native.csr_fingerprint(graph.weights, graph.body_ids)


@dataclass
class BrainInfo:
    """Summary of a loaded substrate (identity, size, attachments)."""

    substrate_id: str
    fingerprint: str
    n_neurons: int
    n_edges: int
    has_annotations: bool
    has_neurotransmitters: bool
    has_receptors: bool
    native_backend: str

    def summary(self) -> str:
        lines = [
            f"Substrate:   {self.substrate_id}",
            f"Fingerprint: {self.fingerprint[:16]}...",
            f"Neurons:     {self.n_neurons:,}",
            f"Connections: {self.n_edges:,}",
            f"Annotations: {'yes' if self.has_annotations else 'no'}"
            f"  Neurotransmitters: {'yes' if self.has_neurotransmitters else 'no'}"
            f"  Receptors: {'yes' if self.has_receptors else 'no'}",
            f"Native core: {self.native_backend}",
        ]
        return "\n".join(lines)


@dataclass
class BiologicalBrain:
    graph: object
    annotations: Path | None = None
    neurotransmitters: Path | None = None
    receptors: Path | None = None
    _metadata_store: object | None = None

    @property
    def metadata(self):
        from .metadata import NeuronMetadataStore
        if self._metadata_store is None:
            self._metadata_store = NeuronMetadataStore(self.annotations)
        return self._metadata_store

    @property
    def n_neurons(self):
        return self.graph.n_neurons

    @property
    def substrate_id(self) -> str:
        return SUBSTRATE_ID

    @property
    def fingerprint_obj(self):
        from ..data.fingerprint import compute_fingerprint
        return compute_fingerprint(self.graph)

    @property
    def fingerprint(self) -> str:
        from ..data.fingerprint import compute_fingerprint
        return compute_fingerprint(self.graph).content_hash

    def info(self) -> BrainInfo:
        import axonweave as _ax
        return BrainInfo(
            substrate_id=self.substrate_id,
            fingerprint=self.fingerprint,
            n_neurons=self.graph.n_neurons,
            n_edges=self.graph.n_edges,
            has_annotations=self.annotations is not None,
            has_neurotransmitters=self.neurotransmitters is not None,
            has_receptors=self.receptors is not None,
            native_backend=_ax.native_version(),
        )

    def capabilities(self) -> dict:
        """Machine-readable capability map of this brain instance."""
        return {
            "substrate_id": self.substrate_id,
            "fingerprint": self.fingerprint,
            "n_neurons": self.graph.n_neurons,
            "n_edges": self.graph.n_edges,
            "backends": {
                "numpy": True,
                "torch": True,
                "keras": True,
            },
            "facades": ["layer", "torch_layer", "keras_layer",
                        "task", "agent", "simulate", "experiment"],
            "dynamics": ["rate", "lif", "adaptive_lif"],
            "learning": ["stdp", "dopamine_stdp"],
            "attachments": {
                "annotations": self.annotations is not None,
                "neurotransmitters": self.neurotransmitters is not None,
                "receptors": self.receptors is not None,
            },
        }

    def torch_layer(self, **kwargs):
        from ..torch import ConnectomeLayer
        return ConnectomeLayer(self.graph, **kwargs)

    def keras_layer(self, **kwargs):
        from ..keras import ConnectomeLayer
        return ConnectomeLayer(self.graph, **kwargs)

    def task(self, **kwargs):
        """Supervised BrainModel facade (Phase 1, requires torch)."""
        from ..brain_api import brain_task
        return brain_task(self, **kwargs)

    def keras_task(self, input=None, output=None, dynamics="rate", selection=None, **kwargs):
        """Supervised Keras BrainLayer facade (requires tensorflow).

        Keras mirror of :meth:`task`: wires Input/Readout interface
        descriptors into a :class:`~axonweave.frameworks.keras.BrainLayer`
        suitable for the Functional/Sequential APIs and ``model.fit()``.
        """
        try:
            import tensorflow  # noqa: F401
        except ImportError as e:
            from ..errors import BackendUnavailableError
            raise BackendUnavailableError(
                "AXW006: brain.keras_task requires TensorFlow; "
                "install axonweave[tensorflow]"
            ) from e
        from ..frameworks.keras import BrainLayer
        from ..frameworks.interfaces import Input, Readout

        model = BrainLayer(self, dynamics=dynamics, selection=selection, **kwargs)
        if input is not None:
            size = getattr(input, "n_target", None) or getattr(input, "size", None)
            if size is None:
                from ..errors import ApiUsageError
                raise ApiUsageError("AXW010: input encoder must expose n_target")
            model.connect(Input(int(size)))
        if output is not None:
            size = getattr(output, "vocab_size", None) or getattr(output, "actions", None)
            if size is None:
                from ..errors import ApiUsageError
                raise ApiUsageError("AXW010: output decoder must expose vocab_size or actions")
            model.connect(Readout(int(size)))
        return model

    def agent(self, **kwargs):
        """Environment agent facade (Phase 1/5)."""
        from ..brain_api import brain_agent
        return brain_agent(self, **kwargs)

    def layer(self, **kwargs):
        """Backward-compatible alias for torch_layer."""
        return self.torch_layer(**kwargs)

    def simulate(self, environment, **kwargs):
        """Simulation-style run (no learning) via the experiment loop."""
        agent = self.agent(**kwargs)
        return agent.run(environment, learn=False, **{
            k: v for k, v in kwargs.items() if k in ("episodes", "max_steps")})

    def experiment(self, environment, **kwargs):
        """Benchmark-style run (learning enabled) via the experiment loop."""
        agent = self.agent(**{
            k: v for k, v in kwargs.items() if k != "episodes"})
        return agent.run(environment, **{
            k: v for k, v in kwargs.items() if k in ("episodes", "max_steps", "log_dir")})
