__version__ = "0.2.0"

from .core.graph import ConnectomeGraph
from .core.brain import BiologicalBrain
from .core.metadata import NeuronMetadata, NeuronMetadataStore
from .data.registry import SubstrateRegistry
from .readout import (
    ActionReadout,
    ClassificationReadout,
    RegressionReadout,
    TokenReadout,
)
from .signals import (
    SignalPolicy,
    NeurotransmitterGain,
    LeakyPropagation,
    NeurotransmitterType,
    SynapseNeurotransmitterModel,
)
from .delays import SynapticDelayEngine, UniformDelay, FixedDelay, NormalDelay
from .receptors import (
    ReceptorModel,
    AMPAReceptor,
    GABAReceptor,
    NMDAReceptor,
    DopamineReceptor,
    ReceptorPolicy,
)
from .training import MixedPrecisionPolicy, DistributedPartitioner

try:
    from ._native import version as native_version
except ImportError:
    def native_version():
        return "python-fallback"

def load(name="male-cns:v1.0"):
    """Load a provisioned substrate from the local AxonWeave cache."""
    return SubstrateRegistry().load(name)

__all__ = [
    "ConnectomeGraph", "BiologicalBrain", "NeuronMetadata", "NeuronMetadataStore",
    "SubstrateRegistry", "SignalPolicy",
    "NeurotransmitterGain", "LeakyPropagation", "NeurotransmitterType",
    "SynapseNeurotransmitterModel", "SynapticDelayEngine",
    "UniformDelay", "FixedDelay", "NormalDelay",
    "load", "native_version",
    "ClassificationReadout", "RegressionReadout", "TokenReadout", "ActionReadout",
    "ReceptorModel", "AMPAReceptor", "GABAReceptor", "NMDAReceptor",
    "DopamineReceptor", "ReceptorPolicy",
    "MixedPrecisionPolicy", "DistributedPartitioner",
]
