"""
This module contains the methods and functions that are utilized to transform the raw audio signal
into a set of features which can be used downstream for statistical analysis/machine learning.
"""

import logging
from math import floor
from typing import Callable

import numpy as np
import scipy.signal as sig

from common import Audio, Frame

WINDOW_FN: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "hamming": lambda x: np.hamming(len(x)) * x
}

DTYPE = np.float32

log = logging.getLogger(__name__)


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
    index = 0

    while hop_start < audio.sample_count:
        hop_end = hop_start + hop_samples
        frame_end = hop_start + frame_samples
        waveform = audio.waveform[hop_start:frame_end]

        # Zero-pad the last frame if the signal doesn't divide evenly.
        if len(waveform) < frame_samples:
            pad_length = frame_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad_length))
            log.debug(
                "Zero padded frame[%d] with %d additional zero samples",
                index,
                pad_length,
            )

        frame_audio = Audio(WINDOW_FN[window_fn](waveform), audio.sample_rate)
        frames.append(Frame(frame_audio, hop_start, frame_end - 1))
        hop_start = hop_end
        index += 1

    log.debug(
        "Split audio into %d frames (hop=%dms,frame=%dms)",
        len(frames),
        hop_ms,
        frame_ms,
    )

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

    log.debug("Emphasized high-frequency components (α=%f)", alpha)

    return Audio(emphasized.astype(DTYPE), audio.sample_rate)  # pyright: ignore[reportAttributeAccessIssue]


def zero_crossing_rate(frame: Frame) -> np.ndarray:
    """
    The zero-crossing rate (ZCR). It is a fraction of consecutive sample pairs that
    change sign. Returns a (1,) array indicating a 1-dimensional feature vector.
    """
    x = frame.audio.waveform
    N = frame.audio.sample_count

    signs = np.sign(x)
    crossings = np.sum(signs[:-1] != signs[1])
    zcr = crossings / (N - 1)

    return np.array([zcr], dtype=DTYPE)


def rms_energy(frame: Frame) -> np.ndarray:
    """
    The root mean square energy of the frame.
    Returns a (1,) array indicating a 1-dimensional feture vector
    """
    x = frame.audio.waveform
    rms = np.sqrt(np.mean(x**2))
    return np.array([rms], dtype=DTYPE)


def power_spectrum(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectrum of a frame.

    Returns
    -------
    freqs : np.ndarray, shape (M,)
        Centre frequency of each bin in Hz.
    power : np.ndarray, shape (M,)
        Power (magnitude squared) at each frequency bin.
    """
    x = frame.audio.waveform
    N = frame.audio.sample_count
    fs = frame.audio.sample_rate

    fft_result = np.fft.rfft(x)  # shape: (N//2 + 1,), reminder: N//2 is frequency bins
    power = (np.abs(fft_result) ** 2) / N  # normalise by frame length
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)  # frequency label for each bin

    return freqs, power


def mel_filterbank(
    freqs: np.ndarray,
    n_filters: int = 26,
    sr: int = 16_000,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """
    Build a Mel filterbank matrix. (speech book ch.4)

    Parameters
    ----------
    freqs : np.ndarray, shape (M,)
        Frequency of each FFT bin in Hz, as returned by power_spectrum.
    n_filters : int
        Number of triangular Mel filters (default: 26).
    sr : int
        Sample rate — used to set f_max if not provided.
    f_min : float
        Lowest frequency covered by the filterbank (Hz).
    f_max : float
        Highest frequency covered by the filterbank (Hz).
        Defaults to the Nyquist frequency (sr / 2).

    Returns
    -------
    filterbank : np.ndarray, shape (N_filters, M)
        Each row is one triangular filter over the FFT bins.
    """
    if f_max is None:
        f_max = sr / 2.0

    # Convert the Hz boundaries to Mel
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)

    # n_filters + 2 points: the two outer edges plus one centre per filter
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)  # back to Hz for comparison with frequencies

    # Build each triangular filter
    filterbank = np.zeros((n_filters, len(freqs)), dtype=DTYPE)

    for i in range(n_filters):
        left = hz_points[i]  # rising edge starts here
        centre = hz_points[i + 1]  # peak
        right = hz_points[i + 2]  # falling edge ends here

        rising = (freqs - left) / (centre - left)
        falling = (right - freqs) / (right - centre)

        # The filter is the elementwise minimum of the two ramps, clipped to [0, 1]
        filterbank[i] = np.maximum(0, np.minimum(rising, falling))

    return filterbank


def _hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def dct(log_energies: np.ndarray, n_coeffs: int = 13) -> np.ndarray:
    """
    Apply a DCT-II to the log filterbank energies and return
    the first n_coeffs cepstral coefficients.

    Parameters
    ----------
    log_energies : np.ndarray, shape (N_filters,)
        Log Mel filterbank energy vector.
    n_coeffs : int
        Number of cepstral coefficients to return (default: 13).

    Returns
    -------
    coeffs : np.ndarray, shape (N_coeffs,)
        The raw MFCC vector. This describes the spectral envelope of the frame.
    """
    M = len(log_energies)
    # Build the DCT-II matrix explicitly — shape (M, M)
    n = np.arange(M)  # filter indices 0..M-1
    k = np.arange(n_coeffs).reshape(-1, 1)
    dct_matrix = np.cos(np.pi * k * (n + 0.5) / M)  # shape (n_coeffs, M)
    return (dct_matrix @ log_energies).astype(DTYPE)  # no slicing needed


def delta(coeffs_matrix: np.ndarray, N: int = 2) -> np.ndarray:
    """
    Compute Δ (first derivative) features across a sequence of frames.

    Parameters
    ----------
    coeffs_matrix : np.ndarray, shape (F, n_coeffs)
        MFCCs for all F frames, one row per frame.
    N : int
        Half-window size for the regression (default: 2).

    Returns
    -------
    deltas : np.ndarray, shape (F, n_coeffs)
    """
    denominator = 2 * np.sum(np.arange(1, N + 1) ** 2)
    numerator = np.zeros_like(coeffs_matrix)

    for n in range(1, N + 1):
        forward = np.concatenate([coeffs_matrix[n:], coeffs_matrix[[-1] * n]])
        backward = np.concatenate([coeffs_matrix[[0] * n], coeffs_matrix[:-n]])
        numerator += n * (forward - backward)

    return (numerator / denominator).astype(DTYPE)


def delta_delta(coeffs_matrix: np.ndarray, N: int = 2) -> np.ndarray:
    """
    Compute ΔΔ (second derivative) features.
    """
    return delta(delta(coeffs_matrix, N), N)


def extract_mfcc(
    frames: list[Frame],
    n_filters: int = 26,
    n_coeffs: int = 13,
    N: int = 2,
) -> np.ndarray:
    """
    Extract MFCCs + Δ + ΔΔ for a list of frames.

    Returns
    -------
    features : np.ndarray, shape (N_frames, n_coeffs * 3)
        39-dimensional feature vector per frame.

    Sources
    -------
    - Mel-frequency Cepstrum | Wikipedia Contibutors (https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
    - Comparaative Evaluation of Various MFCC Implementations on the Speaker Verification Task | T.Ganchev, N.Fakotakis, G.Kokkinakis
    """
    if len(frames) == 0:
        return np.array([])

    mfccs = []
    freqs, _ = power_spectrum(frames[0])
    filterbank = mel_filterbank(freqs, n_filters, frames[0].audio.sample_rate)

    # The per-frame extraction
    for frame in frames:
        _, power = power_spectrum(frame)
        filter_energies = filterbank @ power
        log_energies = np.log(filter_energies + 1e-10)
        coeffs = dct(log_energies, n_coeffs)
        mfccs.append(coeffs)

    mfccs = np.array(mfccs)  # (N_frames, 13)
    deltas = delta(mfccs, N)  # (N_frames, 13)
    ddeltas = delta_delta(mfccs, N)  # (N_frames, 13)

    return np.concatenate([mfccs, deltas, ddeltas], axis=1)  # (N_frames, 39)


def spectral_entropy(power: np.ndarray) -> np.ndarray:
    """Entropy of the power spectrum — lower for speech, higher for noise."""
    total = np.sum(power) + 1e-10
    p = power / total
    entropy = -np.sum(p * np.log(p + 1e-10))
    return np.array([entropy], dtype=DTYPE)


def extract_spectral_features(
    frames: list[Frame],
) -> np.ndarray:
    """
    Extract spectral features for a list of frames.

    Returns
    -------
    features : np.ndarray, shape (N_frames, N_spectral_features)
    """
    features = []

    for frame in frames:
        freqs, power = power_spectrum(frame)

        features.append(np.concatenate([spectral_entropy(freqs, power)]))

    return np.array(features, dtype=np.float32)  # (T, N_spectral_features)


def extract(frames: list[Frame]) -> np.ndarray:
    """
    Extract all features for a list of frames and concatenate into
    a single feature matrix.

    Returns
    -------
    features : np.ndarray, shape (N_frames, N_features)
    """
    log.debug("Extracting features from %d frames", len(frames))

    log.debug("Calculate ZCR")
    zcr = np.array([zero_crossing_rate(f) for f in frames])  # (N_frames, 1)

    log.debug("Calculate RMS Energy")
    rms = np.array([rms_energy(f) for f in frames])  # (N_frames, 1)

    log.debug("Calculate MFCCs")
    mfccs = extract_mfcc(frames)  # (N_frames, N_mfccs)

    log.debug("Calculate spectral features")
    spectral = extract_spectral_features(frames)  # (N_frames, N_spectral_features)

    features = np.concatenate(
        [zcr, rms, mfccs, spectral], axis=1
    )  # (N_frames, N_features)

    log.debug("Created %s feature matrix", features.shape)

    return features


def process(audio: Audio, window_fn: str = "hamming") -> np.ndarray:
    """
    Full audio processing pipeline which starts from raw audio and outputs a feature matrix.

    Parameters
    ----------
    audio : Audio
        Raw mono waveform at the target sample rate.
    window_fn : str
        Window function to apply to each frame (default: "hamming").

    Returns
    ----------
    features : np.ndarray, shape (T, N_total_features)
        One 45-dimensional feature vector per frame.
    """
    audio = pre_emphasis(audio)

    frames = split_into_frames(
        audio,
        frame_ms=25,
        hop_ms=10,
        window_fn=window_fn,
    )

    return extract(frames)
