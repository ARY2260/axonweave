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

    def task(self, **kwargs):
        """Supervised BrainModel facade (Phase 1)."""
        from ..brain_api import brain_task
        return brain_task(self, **kwargs)

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
