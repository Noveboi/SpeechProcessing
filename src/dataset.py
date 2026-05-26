"""
Build the labelled training feature matrix from the speech/ and noise/
directory structure.
"""

import json
import logging
from pathlib import Path

import numpy as np

import extractor
import loader
import preprocessor
from common import Audio, Segment, SegmentLabel

log = logging.getLogger(__name__)

SNR_LEVELS_DB = [0, 5, 10, 15, 20]  # cover the full noise range


def _parse_time(time_str: str) -> float:
    """
    Convert "HH:MM:SS.ss" to seconds.
    """
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_test_transcription(path: str) -> list[Segment] | None:
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


def load_test_audio(test_dir: str) -> list[tuple[Path, Audio]]:
    """
    Loads test audio samples from either:
    - a single WAV file
    - a directory containing WAV files
    """

    path = Path(test_dir)

    if path.is_file():
        wav_files = [path] if path.suffix.lower() == ".wav" else []
    elif path.is_dir():
        wav_files = list(path.glob("**/*.wav"))
    else:
        log.warning("Path does not exist: %s", test_dir)
        return []

    if not wav_files:
        log.warning("No WAV files found!")
        return []

    audio_list: list[tuple[Path, Audio]] = []

    for wav_path in wav_files:
        log.info("Loading test file: %s", wav_path)

        audio = loader.load_audio(str(wav_path))
        audio_list.append((wav_path, audio))

    return audio_list


def _mix_speech_noise(
    speech: np.ndarray, noise: np.ndarray, snr_db: float
) -> np.ndarray:
    """Mix a noise segment into a speech waveform at a target SNR."""
    speech_power = np.mean(speech**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10
    scale = np.sqrt(speech_power / (noise_power * (10 ** (snr_db / 10))))

    # Tile noise to match speech length if needed
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeats)

    return (speech + scale * noise[: len(speech)]).astype(np.float32)


def _build_core(
    speech_dir: str, noise_dir: str
) -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng(1337)

    log.info("Using speech file directory: %s", speech_dir)
    log.info("Using noise file directory: %s", noise_dir)

    speech_files = sorted(Path(speech_dir).glob("**/*.wav"))
    noise_files = sorted(Path(noise_dir).glob("**/*.wav"))

    if len(speech_files) == 0:
        log.critical("No speech files found for training")

    if len(noise_files) == 0:
        log.critical("No noise files found for training")

    if len(speech_files) == 0 or len(noise_files) == 0:
        return None

    # Pre-load all noise waveforms — they're only 6h7m total, fits in memory
    log.info("Pre-loading %d noise files", len(noise_files))
    noise_audio: list[Audio] = []

    for p in noise_files:
        try:
            noise_audio.append(loader.load_audio(str(p)))
        except Exception as e:
            log.warning("Skipping noise file %s — %s", p.name, e)

    X_parts, y_parts = [], []
    count_noise = len(noise_audio)
    count_speech = len(speech_files) * len(SNR_LEVELS_DB)
    counter = 1

    # Speech file processing loop
    for speech_path in speech_files:
        try:
            speech_audio = loader.load_audio(str(speech_path))
        except Exception as e:
            log.error("Failed to load %s — %s", speech_path.name, e)
            continue

        # TYPE 1 - Clean speech (label = 1)
        frames = preprocessor.process(speech_audio)
        features = extractor.extract(frames)
        X_parts.append(features)
        y_parts.append(np.ones(len(features), dtype=np.int8))

        # TYPE 2 - Noisy speech at each SNR level (label = 1)
        for snr_db in SNR_LEVELS_DB:
            noise_idx = rng.integers(count_noise)

            log.info(
                "Building speech features @ %ddB SNR (%d/%d) noise=%d",
                snr_db,
                counter,
                count_speech,
                noise_idx,
            )

            noise_waveform = noise_audio[noise_idx].waveform
            mixed = _mix_speech_noise(speech_audio.waveform, noise_waveform, snr_db)
            mixed_audio = Audio(mixed, speech_audio.sample_rate)

            frames = preprocessor.process(mixed_audio)
            features = extractor.extract(frames)
            X_parts.append(features)
            y_parts.append(np.ones(len(features), dtype=np.int8))

            counter += 1

    # TYPE 3 - Pure noise (label = 0)
    for idx, noise in enumerate(noise_audio):
        log.info("Building noise features (%d/%d)", idx + 1, count_noise)
        frames = preprocessor.process(noise)
        features = extractor.extract(frames)
        X_parts.append(features)
        y_parts.append(np.zeros(len(features), dtype=np.int8))

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)

    log.info(
        "Raw dataset — %d frames  (%d speech / %d noise)",
        len(X),
        np.sum(y == 1),
        np.sum(y == 0),
    )

    # 4. Balance and shuffle
    X, y = _balance(X, y, rng)
    return X, y


def _balance(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Undersample the majority class to match the minority class count.

    This avoids providing the classifiers with 60 hours worth of speech data and 6 hours worth of noise data.
    A balanced set entails fair distribution of data that the classifiers will work with.
    """
    n_speech = np.sum(y == 1)
    n_noise = np.sum(y == 0)
    n_keep = min(n_speech, n_noise)

    speech_idx = np.where(y == 1)[0]
    noise_idx = np.where(y == 0)[0]

    speech_idx = rng.choice(speech_idx, n_keep, replace=False)
    noise_idx = rng.choice(noise_idx, n_keep, replace=False)

    idx = np.concatenate([speech_idx, noise_idx])
    idx = rng.permutation(idx)  # shuffle so classes aren't in blocks

    log.info("Balanced dataset — %d frames per class (%d total)", n_keep, len(idx))
    return X[idx], y[idx]


def create(
    speech_dir: str,
    noise_dir: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Process all WAV files under speech_dir and noise_dir and return a labelled feature matrix.
    This matrix will then be used to train classifiers.

    Labels:
        - 1 = speech (foreground)
        - 0 = noise (background)

    Parameters
    ----------
    speech_dir : str
        Path to the directory containing speech WAV files.
    noise_dir : str
        Path to the directory containing noise WAV files.

    Returns
    -------
    X : np.ndarray, shape (N_frames, N_features)
    y : np.ndarray, shape (N_frames,)
    """

    return _build_core(speech_dir, noise_dir)


# Demo the functionality of dataset.py
if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    from scipy.io import wavfile

    import loader

    argv = sys.argv

    if len(argv) == 1:
        print(f"USAGE: {argv[0]} <COMMAND> [...PARAMETERS]")
        sys.exit(1)

    command = argv[1]
    parameters = argv[2:]

    if command == "mix":
        if len(parameters) < 3:
            print(f"USAGE: {argv[0]} mix <SPEECH_FILE> <SNR_DB> [NOISE_FILE]")
            sys.exit(1)

        speech_path, noise_path, snr_db = parameters
        speech = loader.load_audio(speech_path)
        noise = loader.load_audio(noise_path)
        mixed = _mix_speech_noise(speech.waveform, noise.waveform, float(snr_db))

        window_sec = 0.2
        window_samp = int(window_sec * speech.sample_rate)
        t = np.linspace(0.0, window_sec, window_samp)

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

        ax1.plot(t, speech.waveform[:window_samp])
        ax1.set_title("Original speech (zoomed)")

        ax2.plot(t, mixed[:window_samp])
        ax2.set_title(f"Mixed speech w/ noise @ {snr_db} dB SNR (zoomed)")

        plt.show()

        file_name = f"mixed_{snr_db}db.wav"
        print(f"Saving to {file_name}")
        wavfile.write(file_name, rate=speech.sample_rate, data=mixed)
