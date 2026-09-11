from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Input:
    size: int


@dataclass
class Readout:
    size: int
