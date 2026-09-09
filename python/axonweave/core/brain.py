from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BiologicalBrain:
    graph: object
    annotations: Path | None = None
    neurotransmitters: Path | None = None
    receptors: Path | None = None

    @property
    def n_neurons(self):
        return self.graph.n_neurons

    def torch_layer(self, **kwargs):
        from ..torch import ConnectomeLayer
        return ConnectomeLayer(self.graph, **kwargs)

    def keras_layer(self, **kwargs):
        from ..keras import ConnectomeLayer
        return ConnectomeLayer(self.graph, **kwargs)
