"""
Build the labelled training feature matrix from the speech/ and noise/
directory structure.
"""

import logging
from pathlib import Path

import numpy as np

import loader
import processor
from common import Audio

log = logging.getLogger(__name__)


def load_test_audio(test_dir: str) -> list[tuple[str, Audio]]:
    """
    Loads all the test audio samples.

    Parameters
    --------
    test_dir: str
        The directory path that contains all the test WAV files

    Returns
    --------
    List of tuples containing two elements:
        file_path : str
        audio : Audio
    """
    path = Path(test_dir)
    wav_files = path.glob("**/*.wav")

    if not wav_files:
        log.warning("No WAV files found!")
        return []

    audio_list: list[tuple[str, Audio]] = []

    for path in wav_files:
        log.info("Loading test file: %s", path)
        path_str = str(path)

        audio = loader.load_audio(path_str)
        audio_list.append((path_str, audio))

    return audio_list


def _build_core(speech_dir: str, noise_dir: str) -> tuple[np.ndarray, np.ndarray]:
    X_parts, y_parts = [], []

    for label, directory in [(1, speech_dir), (0, noise_dir)]:
        class_name = "speech" if label == 1 else "noise"
        wav_files = sorted(Path(directory).glob("**/*.wav"))

        if not wav_files:
            log.warning("No WAV files found in %s", directory)
            continue

        log.info(
            "Processing %d %s files from %s",
            len(wav_files),
            class_name,
            directory,
        )

        for i, path in enumerate(wav_files):
            try:
                audio = loader.load_audio(str(path))
                features = processor.process(audio)  # (T, 45)
                labels = np.full(len(features), label)  # (T,)
                X_parts.append(features)
                y_parts.append(labels)
                log.debug(
                    "[%s %d/%d] %s — %d frames",
                    class_name,
                    i + 1,
                    len(wav_files),
                    path.name,
                    len(features),
                )
            except Exception as e:
                log.error("Failed to process %s — %s", path.name, e)
                continue

        log.info(
            "Finished %s files — %d total frames so far",
            class_name,
            sum(len(x) for x in X_parts),
        )

    if not X_parts:
        raise RuntimeError("No files were successfully processed.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    log.info(
        "Dataset built — %d frames total  (%d speech / %d noise)",
        len(X),
        np.sum(y == 1),
        np.sum(y == 0),
    )

    return X, y


def build(
    speech_dir: str, noise_dir: str, cache_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process all WAV files under speech_dir and noise_dir and return
    a labelled feature matrix.

    Directory structure assumed:
        speech_dir/
            *.wav  (and any subdirectories)
        noise_dir/
            *.wav  (and any subdirectories)

    Labels: 1 = speech (foreground), 0 = noise (background)

    Parameters
    ----------
    speech_dir : str
        Path to the directory containing speech WAV files.
    noise_dir : str
        Path to the directory containing noise WAV files.

    Returns
    -------
    X : np.ndarray, shape (N_total_frames, 45)
    y : np.ndarray, shape (N_total_frames,)
    """

    # Save X, y to train only when we want
    cache_dir = cache_dir or Path("_cache")
    X_dir = cache_dir / "X.npy"
    y_dir = cache_dir / "y.npy"

    try:
        with np.load(X_dir) as X_data:
            X = X_data
            log.info("Loaded X_train data from cache")
        with np.load(y_dir) as y_data:
            y = y_data
            log.info("Loaded y_train data from cache")
    except OSError:
        log.warning("No cached files found, begin training...")
        X, y = _build_core(speech_dir, noise_dir)

        with open(X_dir, mode="wb") as Xf:
            np.save(Xf, X)

        with open(y_dir, mode="wb") as yf:
            np.save(yf, y)

    return X, y
