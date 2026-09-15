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

# Temporal Runtime Alpha (v0.2.0)
from .runtime import (
    ConnectomeRuntime, RuntimeState, NeuronState, SynapticState, PlasticityState,
)
from .dynamics import (
    DynamicsModel, Rate, LIF, AdaptiveLIF, DynamicsPolicy,
    SurrogateLIF, SurrogateAdaptiveLIF,
    SurrogateGradient, SigmoidSurrogate, ATanSurrogate,
    PiecewiseSurrogate, StraightThroughEstimator,
)
from .encoders import (
    VectorEncoder, TimeSeriesEncoder, ImageEncoder, TokenEncoder, SensorEncoder,
)

try:
    from ._native import version as native_version
except ImportError:
    def native_version():
        return "python-fallback"

def load(name="male-cns:v1.0"):
    """Load a provisioned substrate from the local AxonWeave cache."""
    return SubstrateRegistry().load(name)

__all__ = [
    # Core
    "ConnectomeGraph", "BiologicalBrain", "NeuronMetadata", "NeuronMetadataStore",
    "SubstrateRegistry", "load", "native_version",
    # Runtime (Temporal Runtime Alpha)
    "ConnectomeRuntime", "RuntimeState", "NeuronState", "SynapticState",
    "PlasticityState",
    # Dynamics
    "DynamicsModel", "Rate", "LIF", "AdaptiveLIF", "DynamicsPolicy",
    "SurrogateLIF", "SurrogateAdaptiveLIF",
    "SurrogateGradient", "SigmoidSurrogate", "ATanSurrogate",
    "PiecewiseSurrogate", "StraightThroughEstimator",
    # Signals / receptors / delays
    "SignalPolicy", "NeurotransmitterGain", "LeakyPropagation",
    "NeurotransmitterType", "SynapseNeurotransmitterModel",
    "SynapticDelayEngine", "UniformDelay", "FixedDelay", "NormalDelay",
    "ReceptorModel", "AMPAReceptor", "GABAReceptor", "NMDAReceptor",
    "DopamineReceptor", "ReceptorPolicy",
    # Encoders
    "VectorEncoder", "TimeSeriesEncoder", "ImageEncoder",
    "TokenEncoder", "SensorEncoder",
    # Readouts
    "ClassificationReadout", "RegressionReadout", "TokenReadout", "ActionReadout",
    # Training
    "MixedPrecisionPolicy", "DistributedPartitioner",
]
