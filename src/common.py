"""
This module contains common types and utilities used throughout the system.
"""

from dataclasses import dataclass

import numpy as np


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
