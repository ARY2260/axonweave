from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True)
class SignalPolicy:
    """Explicit mapping from source neurotransmitter labels to model gains."""
    mapping: dict[str, float] = field(default_factory=dict)
    default_gain: float = 0.0

    def gain(self, neurotransmitter: str) -> float:
        return float(self.mapping.get(neurotransmitter, self.default_gain))

@dataclass
class NeurotransmitterGain:
    policy: SignalPolicy

    def apply(self, neurotransmitter: str, signal: float) -> float:
        return signal * self.policy.gain(neurotransmitter)

@dataclass
class LeakyPropagation:
    decay: float = 0.95

    def step(self, state, input_signal):
        return self.decay * state + input_signal
