"""
This module contains the methods and functions that are utilized to transform the raw audio signal
into a set of features which can be used downstream for statistical analysis/machine learning.
"""

from dataclasses import dataclass
from math import floor
from typing import Callable

import numpy as np
import scipy.signal as sig

from common import Audio

WINDOW_FN: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "hamming": lambda x: np.hamming(len(x)) * x
}


@dataclass
class Frame:
    audio: Audio
    start_idx: int
    end_idx: int


def split_into_frames(
    audio: Audio,
    frame_ms: int,
    hop_ms: int,
    window_fn: str,
) -> list[Frame]:
    frame_samples = floor((frame_ms * audio.sample_rate) / 1000)
    hop_samples = floor((hop_ms * audio.sample_rate) / 1000)

    frames: list[Frame] = []
    hop_start = 0

    while hop_start < audio.sample_count:
        hop_end = hop_start + hop_samples
        frame_end = hop_start + frame_samples
        waveform = audio.waveform[hop_start:frame_end]

        # Zero-pad the last frame if the signal doesn't divide evenly.
        if len(waveform) < frame_samples:
            waveform = np.pad(waveform, (0, frame_samples - len(waveform)))

        frame_audio = Audio(WINDOW_FN[window_fn](waveform), audio.sample_rate)
        frames.append(Frame(frame_audio, hop_start, frame_end - 1))
        hop_start = hop_end

    return frames


def pre_emphasis(audio: Audio, alpha: float = 0.97) -> Audio:
    """
    This boosts high-frequency content to account for the natural ~6dB/octave
    spectral roll-of the vocal tract. In other words, since the high-frequency components
    of speech are naturally quieter than the lower frequency ones, we 'normalize' the frequency content
    across the entire spectrum for a more balanced speech sample, suitable for processing.

    This 'boost' is implemented as a simple first-order FIR high-pass filter:
        y[n] = x[n] - α * x[n-1]
    """
    # In the ``lfilter`` function below, ``b`` represents the coefficients of the current
    # and past inputs samples, ``a`` represents the coefficients of the current and past output
    # samples. Since we have no feedback, ``a`` is of length 1.
    emphasized = sig.lfilter(b=[1, -alpha], a=[1.0], x=audio.waveform)

    return Audio(emphasized.astype(np.float32), audio.sample_rate)  # pyright: ignore[reportAttributeAccessIssue]


def process(audio: Audio):
    pass
