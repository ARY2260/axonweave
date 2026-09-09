__version__ = "0.1.0"

from .core.graph import ConnectomeGraph
from .core.brain import BiologicalBrain
from .data.registry import SubstrateRegistry
from .readout import (
    ActionReadout,
    ClassificationReadout,
    RegressionReadout,
    TokenReadout,
)
from .signals import SignalPolicy, NeurotransmitterGain, LeakyPropagation

try:
    from ._native import version as native_version
except ImportError:
    def native_version():
        return "python-fallback"

def load(name="male-cns:v1.0"):
    """Load a provisioned substrate from the local AxonWeave cache."""
    return SubstrateRegistry().load(name)

__all__ = [
    "ConnectomeGraph", "BiologicalBrain", "SubstrateRegistry", "SignalPolicy",
    "NeurotransmitterGain", "LeakyPropagation", "load", "native_version",
    "ClassificationReadout", "RegressionReadout", "TokenReadout", "ActionReadout",
]
