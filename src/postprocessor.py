"""
Smooth raw frame-level classifier predictions into clean segment
boundaries and write results to CSV.
"""

import csv
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)


def smooth_predictions(
    predictions: np.ndarray,
    window_ms: int = 300,
    hop_ms: int = 10,
) -> np.ndarray:
    """
    Apply a median filter to the "raw" predictions.

    For binary labels, the median over a window is equivalent to a
    majority vote. In other words, the label held by more than half
    the frames in the window wins.

    DISTINCTION:
        "Window" here refers to the sliding window used in the ``median_filter`` algorithm,
        which has its size defined by the ``window_ms`` parameter. Not to be confused with
        the

    Parameters
    ----------
    predictions : np.ndarray, shape (T,)
        Raw binary predictions from the classifier (0=noise, 1=speech).
    window_ms : int
        Width of the smoothing window in milliseconds (default: 300).
        Should be long enough to absorb isolated wrong predictions but
        short enough not to blur genuine short segments.
    hop_ms : int
        Frame hop size used during feature extraction (default: 10).

    Returns
    -------
    smoothed : np.ndarray, shape (T,)  dtype int
    """
    window_frames = max(1, window_ms // hop_ms)

    # window_frames must be odd for a symmetric window
    if window_frames % 2 == 0:
        window_frames += 1

    smoothed = median_filter(predictions, size=window_frames, mode="nearest")
    smoothed = smoothed.astype(int)

    n_changed = np.sum(predictions != smoothed)

    logger.info(
        "Median filter (window=%d ms / %d frames) — %d frames changed (%.1f%%)",
        window_ms,
        window_frames,
        n_changed,
        100 * n_changed / len(predictions),
    )

    return smoothed
