"""
Build the labelled training feature matrix from the speech/ and noise/
directory structure.
"""

import logging
from pathlib import Path

import numpy as np

import loader
import processor

logger = logging.getLogger(__name__)


def build(
    speech_dir: str,
    noise_dir: str,
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
    X_parts, y_parts = [], []

    for label, directory in [(1, speech_dir), (0, noise_dir)]:
        class_name = "speech" if label == 1 else "noise"
        wav_files = sorted(Path(directory).glob("**/*.wav"))

        if not wav_files:
            logger.warning("No WAV files found in %s", directory)
            continue

        logger.info(
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
                logger.debug(
                    "[%s %d/%d] %s — %d frames",
                    class_name,
                    i + 1,
                    len(wav_files),
                    path.name,
                    len(features),
                )
            except Exception as e:
                logger.error("Failed to process %s — %s", path.name, e)
                continue

        logger.info(
            "Finished %s files — %d total frames so far",
            class_name,
            sum(len(x) for x in X_parts),
        )

    if not X_parts:
        raise RuntimeError("No files were successfully processed.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    logger.info(
        "Dataset built — %d frames total  (%d speech / %d noise)",
        len(X),
        np.sum(y == 1),
        np.sum(y == 0),
    )

    # Save X, y to train only when we want
    # TODO

    return X, y
