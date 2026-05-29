from math import floor
from typing import Callable

import numpy as np
import scipy.signal as sig

from common import Audio, Frame

WINDOW_FN: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "hamming": lambda x: np.hamming(len(x)) * x
}

DTYPE = np.float32


def split_into_frames(
    audio: Audio,
    frame_ms: int,
    hop_ms: int,
    window: str,
) -> list[Frame]:

    if window not in WINDOW_FN:
        raise ValueError(
            f"Unknown window function '{window}'. Available: {list(WINDOW_FN)}"
        )

    window_func = WINDOW_FN[window]

    frame_samples = floor((frame_ms * audio.sample_rate) / 1000)
    hop_samples = floor((hop_ms * audio.sample_rate) / 1000)

    frames: list[Frame] = []
    hop_start = 0
    index = 0

    while hop_start < audio.sample_count:
        hop_end = hop_start + hop_samples
        frame_end = hop_start + frame_samples
        waveform = audio.waveform[hop_start:frame_end]

        # Zero-pad the last frame if the signal doesn't divide evenly.
        if len(waveform) < frame_samples:
            pad_length = frame_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad_length))

        frame_audio = Audio(window_func(waveform), audio.sample_rate)
        frames.append(Frame(frame_audio, hop_start, frame_end - 1))
        hop_start = hop_end
        index += 1

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

    return Audio(emphasized.astype(DTYPE), audio.sample_rate)  # pyright: ignore[reportAttributeAccessIssue]


def remove_dc_offset(audio: Audio) -> Audio:
    """
    Subtract the signal mean to centre the waveform around zero.

    This gives more predictable and normalized/standardized waveforms which benefit the feature extraction process.
    """
    corrected = audio.waveform - np.mean(audio.waveform)
    return Audio(corrected.astype(DTYPE), audio.sample_rate)


def process(
    audio: Audio,
    window_fn: str = "hamming",
    frame_ms: int = 25,
    hop_ms: int = 10,
) -> list[Frame]:
    """
    Parameters
    ----------
    audio : Audio
        Raw mono waveform at the target sample rate.
    window_fn : str
        Window function to apply to each frame (default: "hamming").

    Returns
    ----------
    frames : list[Frame]
        The windowed frames.
    """
    audio = remove_dc_offset(audio)
    audio = pre_emphasis(audio)

    frames = split_into_frames(
        audio,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        window=window_fn,
    )

    return frames
