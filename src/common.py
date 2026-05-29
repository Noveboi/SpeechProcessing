"""
This module contains common types and utilities used throughout the system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Iterator, TypeAlias

import numpy as np

Seconds: TypeAlias = float


class SegmentLabel(Enum):
    BACKGROUND = "background"
    FOREGROUND = "foreground"


@dataclass(frozen=True)
class Audio:
    waveform: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> Seconds:
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
    start: Seconds
    end: Seconds
    label: SegmentLabel

    @property
    def duration(self) -> Seconds:
        return self.end - self.start


def _seg_merge(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: (s.start, s.end))

    merged: list[Segment] = [ordered[0]]

    for seg in ordered[1:]:
        last = merged[-1]

        # Merge only if the label matches and the intervals overlap or touch
        if seg.label == last.label and seg.start <= last.end:
            merged[-1] = Segment(
                start=last.start, end=max(last.end, seg.end), label=last.label
            )
        else:
            merged.append(seg)

    return merged


class SegmentCollection(Iterable):
    """
    A collection of non-overlapping temporal foreground/background segments
    """

    def __init__(self, segments: list[Segment]):
        self.data = _seg_merge(segments)

    def where(self, predicate: Callable[[Segment], bool]) -> "SegmentCollection":
        return SegmentCollection([s for s in self.data if predicate(s)])

    def __iter__(self) -> Iterator[Segment]:
        return self.data.__iter__()

    def __len__(self) -> int:
        return len(self.data)
