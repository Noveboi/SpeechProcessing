"""
This module contains the methods and functions that are utilized to transform the raw audio signal
into a set of features which can be used downstream for statistical analysis/machine learning.
"""

import logging

import numpy as np

from common import Frame

DTYPE = np.float32
log = logging.getLogger(__name__)


def zero_crossing_rate(frame: Frame) -> np.ndarray:
    """
    The zero-crossing rate (ZCR). It is a fraction of consecutive sample pairs that
    change sign. Returns a (1,) array indicating a 1-dimensional feature vector.
    """
    x = frame.audio.waveform
    N = frame.audio.sample_count

    signs = np.sign(x)
    crossings = np.sum(signs[:-1] != signs[1:])
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
    freqs : np.ndarray, shape (N_freqs,)
        Centre frequency of each bin in Hz.
    power : np.ndarray, shape (N_freqs,)
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
    freqs : np.ndarray, shape (N_freqs,)
        Frequency of each FFT bin in Hz, as returned by power_spectrum.
    n_filters : int
        Number of triangular Mel filters.
    sr : int
        Sample rate — used to set f_max if not provided.
    f_min : float
        Lowest frequency covered by the filterbank (Hz).
    f_max : float
        Highest frequency covered by the filterbank (Hz).
        Defaults to the Nyquist frequency (sr / 2).

    Returns
    -------
    filterbank : np.ndarray, (N_filters, N_freqs)
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
    coeffs_matrix : np.ndarray, shape (N_frames, N_coeffs)
        MFCCs for all N_frames frames, one row per frame.
    N : int
        Half-window size for the regression.

    Returns
    -------
    deltas : np.ndarray, (N_frames, N_coeffs)
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


def cmvn(mfccs: np.ndarray) -> np.ndarray:
    """
    Perform 'Cepstral Mean and Variance Normalization' on the raw MFCCs.

    This removes channel and noise-induced bias across speech and is SUPER good
    for speech in noisy environments (such as the one given by the project).
    """
    mean = np.mean(mfccs, axis=0)
    std = np.std(mfccs, axis=0) + 1e-10

    return (mfccs - mean) / std


def spectral_entropy(power: np.ndarray) -> np.ndarray:
    """Entropy of the power spectrum — lower for speech, higher for noise."""
    total = np.sum(power) + 1e-10
    p = power / total
    entropy = -np.sum(p * np.log(p + 1e-10))
    return np.array([entropy], dtype=DTYPE)


def extract(frames: list[Frame]) -> np.ndarray:
    if not frames:
        return np.array([])

    freqs, _ = power_spectrum(frames[0])
    filterbank = mel_filterbank(freqs, sr=frames[0].audio.sample_rate)

    zcr_list, rms_list, mfcc_list, entropy_list = [], [], [], []

    for frame in frames:
        _, power = power_spectrum(frame)

        zcr_list.append(zero_crossing_rate(frame))
        rms_list.append(rms_energy(frame))

        filter_energies = filterbank @ power  # (N_fil, N_freqs) @ (N_freqs,) = (N_fil,)
        log_energies = np.log(filter_energies + 1e-10)  # (N_fil,)
        transformed = dct(log_energies, n=13)  # (13, )
        mfcc_list.append(transformed)

        entropy_list.append(spectral_entropy(power))

    zcr = np.array(zcr_list, dtype=DTYPE)  # (N_frames,)
    rms = np.array(rms_list, dtype=DTYPE)  # (N_frames,)
    entropy = np.array(entropy_list, dtype=DTYPE)  # (N_frames,)
    mfccs = np.array(mfcc_list, dtype=DTYPE)  # (N_frames, 13)
    # mfccs = cmvn(mfccs)  # (N_frames, 13)
    deltas = delta(mfccs)  # (N_frames, 13)
    ddeltas = delta_delta(mfccs)  # (N_frames, 13)
    mfccs_all = np.concatenate([mfccs, deltas, ddeltas], axis=1)  # (N_frames, 39)

    return np.concatenate([zcr, rms, mfccs_all, entropy], axis=1)
