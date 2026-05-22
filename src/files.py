import csv
import logging
from pathlib import Path

from common import Segment, SegmentLabel

log = logging.getLogger(__name__)

CSV_FIELDNAMES = ["Audiofile", "start", "end", "class"]


def read_csv_as_segments(file_path: str) -> list[Segment] | None:
    """
    Read a CSV and translate it into background/foreground segments
    """

    segments: list[Segment] = []

    try:
        with open(file_path, "r", newline="") as f:
            f.readline()  # skip headers
            reader = csv.DictReader(f, fieldnames=CSV_FIELDNAMES)

            for row in reader:
                segments.append(
                    Segment(
                        start=float(row["start"]),
                        end=float(row["end"]),
                        label=SegmentLabel(row["class"]),
                    )
                )
    except OSError as err:
        log.critical("Couldn't read CSV file: %s", err.strerror, exc_info=err)
        return None

    return segments


def write_csv(
    segments: list[Segment],
    audio_filename: str,
    output_path: str,
) -> bool:
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

    try:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
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
    except OSError as err:
        log.critical("Couldn't write CSV file.", exc_info=err)
        return False

    log.info("CSV written → %s  (%d rows)", output_path, len(segments))
    return True
