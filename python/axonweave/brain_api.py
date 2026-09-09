"""High-level facade methods added to BiologicalBrain (Phase 1).

Implemented as a module-level extension function invoked from
``core.brain`` so the dataclass stays pickle-friendly.
"""
from __future__ import annotations


def brain_task(brain, input=None, output=None, dynamics="rate", **kwargs):
    """``brain.task(...)`` — supervised BrainModel with optional interfaces."""
    torch = kwargs.pop("_torch", None)
    if torch is None:
        try:
            import torch  # noqa: F401
        except ImportError as e:
            from .errors import BackendUnavailableError
            raise BackendUnavailableError(
                "AXW006: brain.task requires PyTorch; install axonweave[torch]"
            ) from e
    from .frameworks.torch import BrainModel
    from .encoders import ImageEncoder, SensorEncoder, TokenEncoder
    from .decoders import ActionDecoder, TokenDecoder, ClassificationHead

    model = BrainModel(brain, dynamics=dynamics, **kwargs)
    if input is not None or output is not None:
        from .frameworks.torch import Input, Readout
        if input is not None:
            size = getattr(input, "n_target", None) or getattr(input, "size", None)
            if size is None:
                raise ValueError("AXW010: input encoder must expose n_target")
            model.connect(Input(int(size)))
        if output is not None:
            size = getattr(output, "vocab_size", None) or getattr(output, "actions", None)
            if size is None:
                raise ValueError("AXW010: output decoder must expose vocab_size or actions")
            model.connect(Readout(int(size)))
        model._task_encoder = input
        model._task_decoder = output
    return model


def brain_agent(brain, environment=None, input=None, output=None,
                dynamics="rate", learning=None, **kwargs):
    """``brain.agent(...)`` — environment agent with the standard loop."""
    from .experiment import Agent

    agent = Agent(brain, dynamics=dynamics, learning=learning, **kwargs)
    if input is not None:
        agent.sense(input)
    if output is not None:
        agent.act(output)
    return agent
