import csv
import logging
from pathlib import Path

from common import Segment

log = logging.getLogger(__name__)


def read_csv(file_path: str): ...


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
