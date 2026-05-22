import logging
from dataclasses import dataclass

import files
from common import Seconds, Segment, SegmentLabel

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
    foreground_overlap_percent: float


def evaluate(
    prediction_segments: list[Segment],
    ground_truth_segments: list[Segment],
) -> Evaluation:
    """
    Evaluate the predicted foreground/background segments using the ground truth segments.
    """
    # Compute foreground overlap -> first total duration, percentage after
    fg_pred = [s for s in prediction_segments if s.label == SegmentLabel.FOREGROUND]
    fg_gt = [s for s in ground_truth_segments if s.label == SegmentLabel.FOREGROUND]

    ...

    # return Evaluation(
    #     prediction_stats=statistics(prediction_segments),
    #     ground_truth_stats=statistics(ground_truth_segments),
    # )


def statistics(segments: list[Segment]) -> SegmentStatistics:
    total_duration = sum(s.duration for s in segments)
    fg = [s for s in segments if s.label == SegmentLabel.FOREGROUND]
    bg = [s for s in segments if s.label == SegmentLabel.BACKGROUND]

    fg_count = len(fg)
    bg_count = len(bg)

    fg_total_duration = sum(s.duration for s in fg)
    bg_total_duration = sum(s.duration for s in bg)

    fg_max_duration = max(s.duration for s in fg)
    bg_max_duration = max(s.duration for s in bg)
    fg_min_duration = min(s.duration for s in fg)
    bg_min_duration = min(s.duration for s in bg)

    return SegmentStatistics(
        foreground_percent=fg_total_duration / total_duration,
        background_percent=bg_total_duration / total_duration,
        foreground_average_duration=fg_total_duration / fg_count,
        background_average_duration=bg_total_duration / bg_count,
        foreground_max_duration=fg_max_duration,
        foreground_min_duration=fg_min_duration,
        background_max_duration=bg_max_duration,
        background_min_duration=bg_min_duration,
    )


if __name__ == "__main__":
    segments = files.read_csv_as_segments(
        "results/07_double_layers_full/MLP_s01_full.csv"
    )

    if not segments:
        exit(1)

    stats = statistics(segments)
    print(stats)
