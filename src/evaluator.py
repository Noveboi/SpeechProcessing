import logging
from dataclasses import dataclass

import numpy as np

from common import Seconds, SegmentCollection, SegmentLabel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentStatistics:
    foreground_percent: float
    background_percent: float
    foreground_average_duration: Seconds
    background_average_duration: Seconds
    foreground_max_duration: Seconds
    background_max_duration: Seconds
    foreground_min_duration: Seconds
    background_min_duration: Seconds


@dataclass(frozen=True)
class Evaluation:
    prediction_stats: SegmentStatistics
    ground_truth_stats: SegmentStatistics

    overall_score: float

    # Time-based overlap metrics
    recall_percent: float
    precision_percent: float
    f1_score: float

    # Error breakdown
    false_alarm_rate: float  # noise labeled as speech
    miss_rate: float  # speech labeled as noise

    # Segmentation quality
    segmentation_ratio: float  # predicted fg segments / gt fg segments


def _total_intersection(
    a_segments: SegmentCollection,
    b_segments: SegmentCollection,
) -> float:
    """
    Total duration of time covered by both a and b foreground segments.
    Merges each side first to avoid double-counting.
    """
    total = 0.0

    for a in a_segments:
        for b in b_segments:
            total += max(0.0, min(a.end, b.end) - max(a.start, b.start))

    return total


def _segmentation_score(ratio: float) -> float:
    """
    Converts segmentation ratio to a [0, 1] score.
    A ratio of 1.0 (perfect) scores 1.0.
    Decays symmetrically for both over- and under-segmentation.
    """
    return 1.0 / (1.0 + abs(ratio - 1.0))


def _overall_score(
    f1: float,
    far: float,
    miss_rate: float,
    segmentation_ratio: float,
    weights: tuple[float, float, float, float] = (0.45, 0.25, 0.20, 0.10),
) -> float:
    """
    Weighted geometric mean of the normalised "component" scores.

    Components
    ----------
    f1        -> already in [0, 1], higher is better
    far       -> inverted: (1 - FAR), lower FAR, higher score
    miss_rate -> inverted: (1 - miss_rate), lower miss,  higher score
    seg_ratio -> mapped via _segmentation_score()

    Weights (must sum to 1.0)
    -------
    1. F1 carries the most weight because it already balances precision and recall.
    2. FAR is weighted next — false alarms are the primary failure mode for noisy audio.
    3. Miss rate is separate from F1 because in VAD, missing speech is a more costly
        error than a false alarm in most applications.
    4. Segmentation quality is useful but secondary to raw detection accuracy.
    """
    w_f1, w_far, w_miss, w_seg = weights
    assert abs(sum(weights) - 1.0) < 1e-9, "Weights must sum to 1.0"

    f1_score = f1
    far_score = max(0.0, 1.0 - far)
    miss_score = max(0.0, 1.0 - miss_rate)
    seg_score = _segmentation_score(segmentation_ratio)

    components = [
        (f1_score, w_f1),
        (far_score, w_far),
        (miss_score, w_miss),
        (seg_score, w_seg),
    ]

    log.info(
        "Score components — F1: %.3f (×%.2f)  FAR: %.3f (×%.2f)  "
        "Miss: %.3f (×%.2f)  Seg: %.3f (×%.2f)",
        f1_score,
        w_f1,
        far_score,
        w_far,
        miss_score,
        w_miss,
        seg_score,
        w_seg,
    )

    # Geometric mean: penalises any single component being very low,
    # unlike an arithmetic mean which allows a bad metric to be masked
    # by strong performance elsewhere.
    score = np.prod([score**weight for score, weight in components])

    return float(score)


def evaluate(
    prediction_segments: SegmentCollection,
    ground_truth_segments: SegmentCollection,
) -> Evaluation:
    fg_pred = prediction_segments.where(lambda s: s.label == SegmentLabel.FOREGROUND)
    fg_gt = ground_truth_segments.where(lambda s: s.label == SegmentLabel.FOREGROUND)
    bg_gt = ground_truth_segments.where(lambda s: s.label == SegmentLabel.BACKGROUND)

    gt_fg_total = sum(s.duration for s in fg_gt)
    gt_bg_total = sum(s.duration for s in bg_gt)
    pred_fg_total = sum(s.duration for s in fg_pred)

    intersection = _total_intersection(fg_pred, fg_gt)

    recall = safe_divide(intersection, gt_fg_total)
    precision = safe_divide(intersection, pred_fg_total)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    miss_rate = 1.0 - recall

    false_alarm_duration = pred_fg_total - intersection
    far = safe_divide(false_alarm_duration, gt_bg_total)

    seg_ratio = safe_divide(len(fg_pred), len(fg_gt))

    log.info(
        "Evaluation — Recall: %.1f%%  Precision: %.1f%%  F1: %.3f  "
        "FAR: %.1f%%  Miss: %.1f%%  SegRatio: %.2f",
        recall * 100,
        precision * 100,
        f1,
        far * 100,
        miss_rate * 100,
        seg_ratio,
    )

    score = _overall_score(
        f1=f1,
        far=far,
        miss_rate=miss_rate,
        segmentation_ratio=seg_ratio,
    )

    log.info("Overall score: %.4f", score)

    return Evaluation(
        overall_score=score,
        prediction_stats=statistics(prediction_segments),
        ground_truth_stats=statistics(ground_truth_segments),
        recall_percent=recall * 100,
        precision_percent=precision * 100,
        f1_score=f1,
        false_alarm_rate=far,
        miss_rate=miss_rate,
        segmentation_ratio=seg_ratio,
    )


def safe_divide(a: float, b: float) -> float:
    return a / b if b > 0.0 else 0.0


def statistics(segments: SegmentCollection) -> SegmentStatistics:
    total_duration = sum(s.duration for s in segments)
    fg = segments.where(lambda s: s.label == SegmentLabel.FOREGROUND)
    bg = segments.where(lambda s: s.label == SegmentLabel.BACKGROUND)

    fg_count = len(fg)
    bg_count = len(bg)

    fg_total_duration = sum(s.duration for s in fg)
    bg_total_duration = sum(s.duration for s in bg)

    fg_max_duration = max((s.duration for s in fg), default=0.0)
    bg_max_duration = max((s.duration for s in bg), default=0.0)
    fg_min_duration = min((s.duration for s in fg), default=0.0)
    bg_min_duration = min((s.duration for s in bg), default=0.0)

    return SegmentStatistics(
        foreground_percent=safe_divide(fg_total_duration, total_duration),
        background_percent=safe_divide(bg_total_duration, total_duration),
        foreground_average_duration=safe_divide(fg_total_duration, fg_count),
        background_average_duration=safe_divide(bg_total_duration, bg_count),
        foreground_max_duration=fg_max_duration,
        foreground_min_duration=fg_min_duration,
        background_max_duration=bg_max_duration,
        background_min_duration=bg_min_duration,
    )
