"""Encoders (Phase 2): map external observations into neural input currents.

Encoders are framework-agnostic: they produce NumPy arrays that adapters
convert to backend tensors. They never modify the substrate graph.
"""
from __future__ import annotations

from .image import ImageEncoder
from .token import TokenEncoder
from .sensor import SensorEncoder

__all__ = ["ImageEncoder", "TokenEncoder", "SensorEncoder"]
