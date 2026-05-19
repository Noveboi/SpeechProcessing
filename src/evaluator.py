import logging

log = logging.getLogger(__name__)


def foreground_overlap(
    transcript_segments: list[dict],
    predicted_segments: list[dict],
) -> float:
    """
    Compute the percentage of transcript speech time that the system
    correctly predicted as foreground.

    Parameters
    ----------
    transcript_segments : list of dicts with keys: start, end (seconds)
        Ground-truth speech intervals from the transcription file.
    predicted_segments : list of dicts with keys: start, end, label
        System output from postprocessor.extract_segments().

    Returns
    -------
    recall : float in [0, 1], higher is more accurate
    """
    # Keep only predicted foreground segments
    pred_foreground = [s for s in predicted_segments if s["label"] == "foreground"]

    # Merge transcript segments to avoid double-counting overlapping intervals
    gt_merged = _merge_intervals(transcript_segments)
    gt_total = sum(s["end"] - s["start"] for s in gt_merged)

    if gt_total == 0.0:
        log.warning("Transcript has zero total speech time.")
        return 0.0

    # For each ground-truth interval, sum up how much of it is covered
    # by any predicted foreground interval
    total_overlap = 0.0
    for gt in gt_merged:
        for pred in pred_foreground:
            overlap = max(
                0.0, min(gt["end"], pred["end"]) - max(gt["start"], pred["start"])
            )
            total_overlap += overlap

    recall = total_overlap / gt_total

    log.info(
        "Foreground overlap — %.1f / %.1f seconds = %.2f%%",
        total_overlap,
        gt_total,
        recall * 100,
    )
    return recall


def _merge_intervals(segments: list[dict]) -> list[dict]:
    """
    Merge overlapping or adjacent intervals into a minimal set of
    non-overlapping intervals.

    Handles the case where transcript speakers overlap in time — without
    merging, overlapping ground-truth intervals would cause their shared
    time to be counted twice in gt_total.
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s["start"])
    merged = [sorted_segs[0].copy()]

    for current in sorted_segs[1:]:
        last = merged[-1]
        if current["start"] <= last["end"]:
            # Overlapping or adjacent — extend the current merged interval
            last["end"] = max(last["end"], current["end"])
        else:
            merged.append(current.copy())

    log.debug(
        "_merge_intervals: %d segments → %d after merging",
        len(segments),
        len(merged),
    )
    return merged
