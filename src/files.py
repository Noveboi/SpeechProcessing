import csv
import json
import logging
from pathlib import Path

import loader
from common import Audio, Segment, SegmentLabel

log = logging.getLogger(__name__)

CSV_FIELDNAMES = ["Audiofile", "start", "end", "class"]


def _parse_time(time_str: str) -> float:
    """
    Convert "HH:MM:SS.ss" to seconds.
    """
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _ensure_path_exists(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_existing_path(path: str) -> Path:
    return _ensure_path_exists(to_path(path))


def to_path(path: str) -> Path:
    return Path(path)


def load_transcription(path: Path) -> list[Segment] | None:
    """
    Load the test transcription JSON and return a list of foreground/background segments, exactly like
    the model used for predicte doutputs.
    """
    try:
        with open(path, mode="r") as f:
            transcript = json.load(f)
    except OSError:
        log.warning("Transcription file not found")
        return None

    speech_segments: list[Segment] = [
        Segment(
            start=_parse_time(t["start_time"]),
            end=_parse_time(t["end_time"]),
            label=SegmentLabel.FOREGROUND,
        )
        for t in transcript
    ]

    if len(speech_segments) == 0:
        return []

    all_segments: list[Segment] = []
    current_end: float = -1.0

    for seg_fg in speech_segments:
        if seg_fg.start > current_end:
            all_segments.append(
                Segment(
                    start=current_end,
                    end=seg_fg.start,
                    label=SegmentLabel.BACKGROUND,
                )
            )

        all_segments.append(seg_fg)
        current_end = seg_fg.end

    log.info("Loaded %d transcript speech segments from %s", len(speech_segments), path)
    return all_segments


def load_test_audio(test_dir: Path) -> list[tuple[Path, Audio]]:
    """
    Loads test audio samples from either:
    - a single WAV file
    - a directory containing WAV files
    """

    if test_dir.is_file():
        wav_files = [test_dir] if test_dir.suffix.lower() == ".wav" else []
    elif test_dir.is_dir():
        wav_files = list(test_dir.glob("**/*.wav"))
    else:
        log.warning("Path does not exist: %s", test_dir)
        return []

    if not wav_files:
        log.warning("No WAV files found!")
        return []

    audio_list: list[tuple[Path, Audio]] = []

    for wav_path in wav_files:
        log.info("Loading test file: %s", wav_path)

        audio = loader.load_audio(wav_path)
        audio_list.append((wav_path, audio))

    return audio_list


def load_csv_as_segments(file_path: Path) -> list[Segment] | None:
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
    output_path: Path,
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
    output_path : Path
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
