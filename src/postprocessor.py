"""
Smooth out raw frame-level classifier predictions into clean segment boundaries
"""

import csv
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

log = logging.getLogger(__name__)


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
    smoothed : np.ndarray, shape (T,)  dtype ``int``
    """
    window_frames = max(1, window_ms // hop_ms)

    # window_frames must be odd for a symmetric window
    if window_frames % 2 == 0:
        window_frames += 1

    smoothed = median_filter(predictions, size=window_frames, mode="nearest")
    smoothed = smoothed.astype(int)

    n_changed = np.sum(predictions != smoothed)

    log.info(
        "Median filter (window=%d ms / %d frames) — %d frames changed (%.1f%%)",
        window_ms,
        window_frames,
        n_changed,
        100 * n_changed / len(predictions),
    )

    return smoothed


def remove_short_segments(
    predictions: np.ndarray,
    min_duration_ms: int = 300,
    hop_ms: int = 10,
) -> np.ndarray:
    """
    Merge segments shorter than min_duration_ms into their neighbours.

    Operates by iterating over runs (contiguous same-label stretches).
    A short run is relabelled to match the label of whichever neighbour
    is longer. If both neighbours are equal length the preceding
    neighbour wins.

    Parameters
    ----------
    predictions : np.ndarray, shape (T,)
        Smoothed binary predictions.
    min_duration_ms : int
        Minimum allowed segment duration in milliseconds (default: 300).
    hop_ms : int
        Frame hop size in milliseconds (default: 10).

    Returns
    -------
    cleaned : np.ndarray, shape (T,), dtype int
    """
    min_frames = max(1, min_duration_ms // hop_ms)
    cleaned = predictions.copy()

    changed = True
    while changed:
        changed = False
        runs = _get_runs(cleaned)

        for i, (label, start, end) in enumerate(runs):
            length = end - start  # frames in this run

            if length >= min_frames:
                continue

            # Determine replacement label from longer neighbour
            prev_len = (runs[i - 1][2] - runs[i - 1][1]) if i > 0 else 0
            next_len = (runs[i + 1][2] - runs[i + 1][1]) if i < len(runs) - 1 else 0

            if prev_len == 0 and next_len == 0:
                # Only one run in the entire sequence — nothing to merge into
                break

            replacement = runs[i - 1][0] if prev_len >= next_len else runs[i + 1][0]
            cleaned[start:end] = replacement
            changed = True  # a merge happened — rescan from the top
            break  # runs list is now stale; recompute and retry

    n_changed = np.sum(predictions != cleaned)
    logger.info(
        "Min duration filter (%d ms / %d frames) — %d frames relabelled (%.1f%%)",
        min_duration_ms,
        min_frames,
        n_changed,
        100 * n_changed / len(predictions),
    )
    return cleaned


def _get_runs(
    predictions: np.ndarray,
) -> list[tuple[int, int, int]]:
    """
    Returns
    -------
    runs : list of (label, start_frame, end_frame)
        end_frame is exclusive — the run covers [start_frame, end_frame).
    """
    if len(predictions) == 0:
        return []

    runs: list[tuple[int, int, int]] = []
    label: int = predictions[0]
    start = 0

    for i in range(1, len(predictions)):
        if predictions[i] != label:
            runs.append((int(label), start, i))
            start = i
            label = predictions[i]

    runs.append((int(label), start, len(predictions)))
    return runs


def extract_segments(
    predictions: np.ndarray,
    hop_ms: int = 10,
) -> list[dict]:
    """
    Convert a cleaned frame-label sequence into a list of time segments.

    Parameters
    ----------
    predictions : np.ndarray, shape (T,)
        Fully post-processed binary predictions.
    hop_ms : int
        Frame hop size in milliseconds (default: 10).

    Returns
    -------
    segments : list of dicts with keys: start, end, label
        start and end are in seconds. label is 'foreground' or 'background'.
    """
    runs = _get_runs(predictions)
    hop_secs = hop_ms / 1000.0

    segments = []
    for label, start_frame, end_frame in runs:
        segments.append(
            {
                "start": round(start_frame * hop_secs, 3),
                "end": round(end_frame * hop_secs, 3),
                "label": "foreground" if label == 1 else "background",
            }
        )

    log.info(
        "Extracted %d segments  (%d foreground, %d background)",
        len(segments),
        sum(1 for s in segments if s["label"] == "foreground"),
        sum(1 for s in segments if s["label"] == "background"),
    )
    return segments


def write_csv(
    segments: list[dict],
    audio_filename: str,
    output_path: str,
) -> None:
    """
    Write the segment list to a CSV file in the required format:

        Audiofile, start, end, class

    Parameters
    ----------
    segments : list of dicts
        As returned by extract_segments.
    audio_filename : str
        Name of the source audio file — written into the Audiofile column.
    output_path : str
        Destination path for the CSV file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Audiofile", "start", "end", "class"])
        writer.writeheader()
        for segment in segments:
            writer.writerow(
                {
                    "Audiofile": audio_filename,
                    "start": segment["start"],
                    "end": segment["end"],
                    "class": segment["label"],
                }
            )

    log.info("CSV written → %s  (%d rows)", output_path, len(segments))


def process(
    predictions: np.ndarray,
    audio_filename: str,
    output_path: str,
    smooth_window_ms: int = 300,
    hop_ms: int = 10,
) -> list[dict]:
    """
    Full post-processing pipeline.

    Returns
    -------
    segments : list of dicts (also written to output_path)
    """
    log.info("Post-processing %d frames for '%s'", len(predictions), audio_filename)

    smoothed = smooth_predictions(predictions, smooth_window_ms, hop_ms)
    segments = extract_segments(smoothed, hop_ms)
    write_csv(segments, audio_filename, output_path)

    return segments
