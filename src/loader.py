"""
This module loads a WAV file and converts it to mono and re-samples it
"""

from math import gcd

import numpy as np
import scipy.signal as sig
from scipy.io import wavfile

from common import Audio


def load_audio(filename: str, sample_rate: int = 16_000) -> Audio:
    fs, audio = wavfile.read(filename)

    # Standardize the numeric format of the samples to float32
    audio = _convert_to_float32(audio)

    # Convert to mono by averaging across all channels (if not mono already)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    waveform = _resample(audio, original_fs=fs, target_fs=sample_rate)
    return Audio(waveform, sample_rate)


def _convert_to_float32(audio: np.ndarray) -> np.ndarray:
    dtype = audio.dtype

    if dtype == np.int16:
        return audio.astype(np.float32) / 32_768.0  # 2^15
    elif dtype == np.int32:
        return audio.astype(np.float32) / 2_147_483_648.0  # 2^31
    elif dtype == np.uint8:
        return (audio.astype(np.float32) - 128.0) / 128.0
    elif dtype in (np.float32, np.float64):
        return audio.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV sample dtype: {dtype}")


def _resample(audio: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
    g = gcd(original_fs, target_fs)
    up = target_fs // g
    down = original_fs // g
    return sig.resample_poly(audio, up, down).astype(np.float32)
