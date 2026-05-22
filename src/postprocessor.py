"""
Smooth out raw frame-level classifier predictions into clean segment boundaries
"""

import csv
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from common import Segment, SegmentLabel

log = logging.getLogger(__name__)


def smooth_predictions(
    predictions: np.ndarray,
    hop_ms: int,
    window_ms: int = 300,
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
    predictions : np.ndarray, shape (N_frames,)
        Raw binary predictions from the classifier (0=noise, 1=speech).
    window_ms : int
        Width of the smoothing window in milliseconds (default: 300).
        Should be long enough to absorb isolated wrong predictions but
        short enough not to blur genuine short segments.
    hop_ms : int
        Frame hop size used during feature extraction .

    Returns
    -------
    smoothed : np.ndarray, shape (N_frames,)  dtype ``int``
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


def apply_hangover(
    predictions: np.ndarray,
    hop_ms: int,
    hangover_ms: int = 400,
) -> np.ndarray:
    hangover_frames = hangover_ms // hop_ms
    result = predictions.copy()
    counter = 0
    for i in range(len(result)):
        if predictions[i] == 1:
            counter = hangover_frames
        elif counter > 0:
            result[i] = 1
            counter -= 1
    return result


def remove_short_segments(
    predictions: np.ndarray,
    hop_ms: int,
    min_duration_ms: int = 300,
) -> np.ndarray:
    """
    Merge segments shorter than min_duration_ms into their neighbours.

    Operates by iterating over runs (contiguous same-label stretches).
    A short run is relabelled to match the label of whichever neighbour
    is longer. If both neighbours are equal length the preceding
    neighbour wins.

    Parameters
    ----------
    predictions : np.ndarray, shape (N_frames,)
        Smoothed binary predictions.
    min_duration_ms : int
        Minimum allowed segment duration in milliseconds (default: 300).
    hop_ms : int
        Frame hop size in milliseconds .

    Returns
    -------
    cleaned : np.ndarray, shape (N_frames,), dtype int
    """
    min_frames = max(1, min_duration_ms // hop_ms)
    cleaned = predictions.copy()

    changed = True

    # this guy is a doozie!!!
    while changed:
        changed = False
        runs = _get_runs(cleaned)

        for i, (_, start, end) in enumerate(runs):
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
    log.info(
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
    Loops through the prediction vector ``y`` and concatenates adjacent elements that are equal.

    Example
    --------
    Input: [0, 0, 1, 0, 1, 1, 1, 1, 1]
    Output: [
        (0, 0, 2),
        (1, 2, 3),
        (0, 3, 4),
        (1, 4, 9)
    ]

    Returns
    --------
    runs : list of (label, start_frame, end_frame)
        end_frame is exclusive — the run covers [start_frame, end_frame).
    """
    N_pred = len(predictions)
    if N_pred == 0:
        return []

    runs: list[tuple[int, int, int]] = []
    current_label: int = predictions[0]
    start = 0

    for i in range(1, N_pred):
        if predictions[i] != current_label:
            runs.append((int(current_label), start, i))
            start = i
            current_label = predictions[i]

    runs.append((int(current_label), start, N_pred))
    return runs


def extract_segments(
    predictions: np.ndarray,
    hop_ms: int = 10,
) -> list[Segment]:
    """
    Convert a cleaned frame-label sequence into a list of time segments.

    Parameters
    ----------
    predictions : np.ndarray, shape (N_frames,)
        Fully post-processed binary predictions.
    hop_ms : int
        Frame hop size in milliseconds .
    """
    runs = _get_runs(predictions)
    hop_secs = hop_ms / 1000.0

    segments: list[Segment] = []

    for label, start_frame, end_frame in runs:
        segment = Segment(
            start=round(start_frame * hop_secs, 3),
            end=round(end_frame * hop_secs, 3),
            label=SegmentLabel.FOREGROUND if label == 1 else SegmentLabel.BACKGROUND,
        )

        segments.append(segment)

    log.info(
        "Extracted %d segments  (%d foreground, %d background)",
        len(segments),
        sum(1 for s in segments if s.label == SegmentLabel.FOREGROUND),
        sum(1 for s in segments if s.label == SegmentLabel.BACKGROUND),
    )
    return segments


def write_csv(
    segments: list[Segment],
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
                    "start": segment.start,
                    "end": segment.end,
                    "class": segment.label.value,
                }
            )

    log.info("CSV written → %s  (%d rows)", output_path, len(segments))


def process(
    predictions: np.ndarray,
    hop_ms: int,
    smooth_window_ms: int = 300,
) -> list[Segment]:
    """
    Full post-processing pipeline.

    Returns
    -------
    segments : list of ``Segments``
    """
    log.info("Post-processing %d frames", len(predictions))

    predictions = smooth_predictions(
        predictions, hop_ms=hop_ms, window_ms=smooth_window_ms
    )
    predictions = apply_hangover(predictions, hop_ms=hop_ms)
    predictions = remove_short_segments(predictions, hop_ms=hop_ms)
    segments = extract_segments(predictions, hop_ms=hop_ms)

    return segments
