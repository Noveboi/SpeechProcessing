"""

!!! DISCLAIMER
This module has been entirely created with AI with certain correction by me to help
quickly assess the feature extraction process visually.
!!!

This module will also be used in the documentation of the system to help visualize certain key points.

BEGIN AI SECTION
Sanity-check plots for every stage of the audio processing pipeline.

Each function is self-contained — it receives only the data it needs,
creates a figure, and either saves it to `output_path` or calls
plt.show() if no path is given.

Usage example
-------------
    import loader, processor, visualizer
    from processor import power_spectrum, mel_filterbank

    audio_raw       = loader.load_audio("file.wav")
    audio_emphasis  = processor.pre_emphasis(audio_raw)
    frames          = processor.split_into_frames(audio_emphasis, 25, 10, "hamming")
    features        = processor.extract(frames)
    mfcc_matrix     = features[:, 2:41]     # columns 2–40

    visualizer.plot_waveform(audio_raw, audio_emphasis)
    visualizer.plot_power_spectrum(frames[50])
    visualizer.plot_mel_filterbank(frames[0])
    visualizer.plot_spectrogram(frames)
    visualizer.plot_mfccs(mfcc_matrix)
    visualizer.plot_time_domain_features(features)
    visualizer.plot_feature_distributions(X_train, y_train)
"""

import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from common import Audio
from processor import Frame, mel_filterbank, power_spectrum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

_CMAP_FEATURES = "inferno"
_CMAP_SPECTRUM = "magma"
_COLOR_SPEECH = "#4C9BE8"
_COLOR_NOISE = "#E8754C"
_COLOR_BEFORE = "#888888"
_COLOR_AFTER = "#4C9BE8"
_FIGSIZE_WIDE = (12, 4)
_FIGSIZE_TALL = (12, 7)
_FIGSIZE_SQUARE = (8, 6)


def _save_or_show(fig: Figure, output_path: str | None) -> None:
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot → %s", output_path)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 1. Waveform — raw vs pre-emphasised
# ---------------------------------------------------------------------------


def plot_waveform(
    audio_raw: Audio,
    audio_emphasized: Audio | None = None,
    output_path: str | None = None,
) -> None:
    """
    Plot the raw waveform and, optionally, the pre-emphasised waveform
    side by side for direct comparison.

    Answers: Did the audio load correctly? Is pre-emphasis doing anything
    visible — i.e. are sharp transients more pronounced after filtering?
    """
    sr = audio_raw.sample_rate
    n = len(audio_raw.waveform)
    t = np.arange(n) / sr

    n_plots = 2 if audio_emphasized is not None else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    # Raw waveform
    axes[0].plot(t, audio_raw.waveform, color=_COLOR_BEFORE, linewidth=0.5)
    axes[0].set_title("Raw waveform")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].axhline(0, color="white", linewidth=0.4, alpha=0.4)
    axes[0].grid(True, alpha=0.2)

    # Pre-emphasised waveform
    if audio_emphasized is not None:
        axes[1].plot(t, audio_emphasized.waveform, color=_COLOR_AFTER, linewidth=0.5)
        axes[1].set_title("Pre-emphasised waveform  (α = 0.97)")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_ylim(-1.1, 1.1)
        axes[1].axhline(0, color="white", linewidth=0.4, alpha=0.4)
        axes[1].grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Waveform  —  {sr} Hz  ·  {n / sr:.2f} s  ·  {n:,} samples", fontsize=11
    )
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Power spectrum — single frame
# ---------------------------------------------------------------------------


def plot_power_spectrum(
    frame: Frame,
    frame_idx: int = 0,
    output_path: str | None = None,
) -> None:
    """
    Plot the power spectrum of a single frame on a linear frequency axis.

    Answers: Does the FFT produce a sensible result — is energy concentrated
    in the expected frequency range (< 8 kHz for speech)?
    """
    freqs, power = power_spectrum(frame)
    power_db = 10 * np.log10(power + 1e-10)  # convert to dB for readability

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(freqs / 1000, power_db, color=_COLOR_AFTER, linewidth=0.8)
    ax.fill_between(
        freqs / 1000, power_db, power_db.min(), alpha=0.15, color=_COLOR_AFTER
    )
    ax.set_title(f"Power spectrum — frame {frame_idx}")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_xlim(0, frame.audio.sample_rate / 2000)
    ax.grid(True, alpha=0.2)
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 3. Mel filterbank
# ---------------------------------------------------------------------------


def plot_mel_filterbank(
    frame: Frame,
    n_filters: int = 26,
    output_path: str | None = None,
) -> None:
    """
    Plot all Mel triangular filters overlaid on the power spectrum of
    a representative frame.

    Answers: Are the filters triangular? Are they narrower at low
    frequencies and wider at high frequencies (as expected from Mel
    spacing)? Does the filterbank cover the full frequency range?
    """
    freqs, power = power_spectrum(frame)
    power_db = 10 * np.log10(power + 1e-10)
    fb = mel_filterbank(freqs, n_filters, frame.audio.sample_rate)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top: filterbank shapes
    cmap = plt.get_cmap("plasma", n_filters)
    for i in range(n_filters):
        axes[0].plot(freqs / 1000, fb[i], color=cmap(i), linewidth=1.0, alpha=0.8)
    axes[0].set_title(f"Mel filterbank — {n_filters} triangular filters")
    axes[0].set_ylabel("Filter weight")
    axes[0].grid(True, alpha=0.2)

    # Bottom: power spectrum with filterbank overlaid (scaled for visibility)
    axes[1].plot(
        freqs / 1000,
        power_db,
        color=_COLOR_BEFORE,
        linewidth=0.7,
        label="Power spectrum (dB)",
        zorder=2,
    )
    scale = power_db.max() - power_db.min()
    for i in range(n_filters):
        axes[1].fill_between(
            freqs / 1000,
            fb[i] * scale + power_db.min(),
            power_db.min(),
            alpha=0.07,
            color=cmap(i),
        )
    axes[1].set_title("Filterbank overlaid on power spectrum")
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Power (dB)")
    axes[1].set_xlim(0, frame.audio.sample_rate / 2000)
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(fontsize=9)

    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 4. Spectrogram — power over all frames
# ---------------------------------------------------------------------------


def plot_spectrogram(
    frames: list[Frame],
    hop_ms: int = 10,
    output_path: str | None = None,
) -> None:
    """
    Build a spectrogram matrix from the per-frame power spectra and
    display it as a heatmap (time × frequency).

    Answers: Does energy evolve over time in a way that matches the
    audio content — e.g. are there clear silent regions and active
    speech regions?
    """
    all_rows = []

    for frame in frames:
        _, power = power_spectrum(frame)
        power_db = 10 * np.log10(power + 1e-10)
        all_rows.append(power_db)

    spectrogram = np.array(all_rows).T  # (freq_bins, T)
    T = spectrogram.shape[1]
    freqs, _ = power_spectrum(frames[0])

    time_axis = np.arange(T) * (hop_ms / 1000)
    freq_axis = freqs / 1000  # kHz

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    img = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        extent=(time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]),
        cmap=_CMAP_SPECTRUM,
        interpolation="nearest",
    )
    plt.colorbar(img, ax=ax, label="Power (dB)", pad=0.01)
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 5. MFCC heatmaps — static, Δ, ΔΔ
# ---------------------------------------------------------------------------


def plot_mfccs(
    mfcc_matrix: np.ndarray,
    hop_ms: int = 10,
    output_path: str | None = None,
) -> None:
    """
    Plot the MFCC, Δ, and ΔΔ matrices as three stacked heatmaps.

    Parameters
    ----------
    mfcc_matrix : np.ndarray, shape (T, 39)
        The full 39-column slice of the feature matrix (columns 2:41).

    Answers: Do the MFCCs vary meaningfully over time? Do the delta
    features show sharp transitions at phoneme boundaries? Is ΔΔ
    peaking at points of rapid spectral change?
    """
    assert mfcc_matrix.shape[1] == 39, "Expected 39 columns (13 MFCCs + 13 Δ + 13 ΔΔ)"

    T = mfcc_matrix.shape[0]
    time_axis = np.arange(T) * (hop_ms / 1000)

    static = mfcc_matrix[:, 0:13].T
    delta = mfcc_matrix[:, 13:26].T
    ddelta = mfcc_matrix[:, 26:39].T

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(3, 1, figure=fig, hspace=0.45)

    panels = [
        (static, "MFCCs (static)"),
        (delta, "Δ MFCCs (velocity)"),
        (ddelta, "ΔΔ MFCCs (acceleration)"),
    ]

    for i, (matrix, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        img = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            extent=(time_axis[0], time_axis[-1], 0.5, 13.5),
            cmap=_CMAP_FEATURES,
            interpolation="nearest",
        )
        plt.colorbar(img, ax=ax, pad=0.01)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Coefficient")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        if i == 2:
            ax.set_xlabel("Time (s)")

    fig.suptitle("MFCC feature matrices", fontsize=12, y=1.01)
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 6. Time-domain features over time — ZCR and RMS
# ---------------------------------------------------------------------------


def plot_time_domain_features(
    features: np.ndarray,
    hop_ms: int = 10,
    output_path: str | None = None,
) -> None:
    """
    Plot ZCR (col 0) and RMS (col 1) across all frames.

    Answers: Do ZCR and RMS track speech activity — i.e. do they rise
    during speech and fall during silence or background noise?
    """
    T = features.shape[0]
    time_axis = np.arange(T) * (hop_ms / 1000)
    zcr = features[:, 0]
    rms = features[:, 1]

    fig, axes = plt.subplots(2, 1, figsize=_FIGSIZE_TALL, sharex=True)

    axes[0].plot(time_axis, zcr, color=_COLOR_AFTER, linewidth=0.8)
    axes[0].fill_between(time_axis, zcr, alpha=0.15, color=_COLOR_AFTER)
    axes[0].set_title("Zero Crossing Rate")
    axes[0].set_ylabel("ZCR")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(time_axis, rms, color=_COLOR_NOISE, linewidth=0.8)
    axes[1].fill_between(time_axis, rms, alpha=0.15, color=_COLOR_NOISE)
    axes[1].set_title("RMS Energy")
    axes[1].set_ylabel("RMS")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.2)

    fig.suptitle("Time-domain features", fontsize=12)
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 7. Feature distributions — speech vs noise
# ---------------------------------------------------------------------------


def plot_feature_distributions(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    output_path: str | None = None,
) -> None:
    """
    Plot per-feature histograms for speech (y=1) and noise (y=0) frames,
    overlaid for direct comparison.

    Only the first 8 features are shown to keep the plot readable —
    ZCR, RMS, and the first 6 MFCCs.

    Answers: Are the speech and noise distributions well-separated for
    any feature? Overlapping distributions indicate a feature that is
    less discriminative on its own.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Full feature matrix.
    y : np.ndarray, shape (N,)
        Binary labels — 1 = speech, 0 = noise.
    """
    if feature_names is None:
        feature_names = ["ZCR", "RMS"] + [f"MFCC {i}" for i in range(1, 7)]

    n_features = min(8, X.shape[1])
    X_speech = X[y == 1]
    X_noise = X[y == 0]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        ax.hist(
            X_speech[:, i],
            bins=60,
            alpha=0.6,
            color=_COLOR_SPEECH,
            label="Speech",
            density=True,
        )
        ax.hist(
            X_noise[:, i],
            bins=60,
            alpha=0.6,
            color=_COLOR_NOISE,
            label="Noise",
            density=True,
        )
        ax.set_title(feature_names[i], fontsize=9)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"Feature distributions — speech ({len(X_speech):,} frames) "
        f"vs noise ({len(X_noise):,} frames)",
        fontsize=11,
    )
    _save_or_show(fig, output_path)
