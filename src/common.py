"""
This module contains common types and utilities used throughout the system.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SegmentLabel(Enum):
    BACKGROUND = "background"
    FOREGROUND = "foreground"


@dataclass(frozen=True)
class Audio:
    waveform: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.sample_count / self.sample_rate

    @property
    def sample_count(self) -> int:
        return len(self.waveform)


@dataclass(frozen=True)
class Frame:
    audio: Audio
    start_idx: int
    end_idx: int


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    label: SegmentLabel
