"""Interface descriptors for BrainModel.connect()."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Input:
    """Declares the input interface dimension (encoder output size)."""
    size: int


@dataclass
class Readout:
    """Declares the readout interface dimension (target classes/tokens)."""
    size: int
