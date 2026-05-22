import logging
import sys
from typing import Any

import numpy as np

import cache
import classifier
import configuration
import dataset
import evaluator
import extractor
import loader
import postprocessor
import preprocessor
from common import Audio, Segment

log = logging.getLogger(__name__)


def train(
    model_name: str, speech_dir: str, noise_dir: str
) -> classifier.FrameClassifier:
    """
    Train the classifier on speech/noise data.

    Returns
    --------
    The trained classifier, ready to make predictions.
    """
    clf = classifier.get(model_name)
    clf_file_path = f"{clf.name}.pkl"

    try:
        clf = cache.load(clf_file_path)
    except OSError:
        log.info("No persisted model found for %s at '%s'", clf.name, clf_file_path)

        data = dataset.build(
            speech_dir=speech_dir,
            noise_dir=noise_dir,
        )

        if not data:
            log.fatal("Training could not finish")
            sys.exit(1)

        X_train, y_train = data

        clf.fit(X_train, y_train)
        cache.dump(clf, clf_file_path)

    return clf


def predict(
    audio: Audio,
    model: classifier.FrameClassifier,
    frame_ms: int = 25,
    hop_ms: int = 10,
) -> list[Segment]:
    frames = preprocessor.process(audio, frame_ms=frame_ms, hop_ms=hop_ms)
    features: np.ndarray = extractor.extract(frames)  # (N_frames, N_features)
    predictions = model.predict(features)  # (N_frames,)
    segments = postprocessor.process(predictions, hop_ms=hop_ms)

    return segments


def test_multiple_audio(test_dir: str, model: classifier.FrameClassifier) -> None:
    """
    Predict foreground/background intervals in audio files contained in the ``test_dir`` directory.

    Parameters
    --------
    test_dir : str
        The directory containing the WAV audio files
    model : FrameClassifier
        The classifier to be used in the predictions/classifications (e.g: ``KNN`` or ``MLP``)

    Side Effects
    --------
    After the predictions/classifications, the results are stored in CSV files.
    """
    for path, audio in dataset.load_test_audio(test_dir):
        log.info("Processing and predicting audio for '%s'", path)

        file_path = path.name
        csv_path = f"results/{model.name}_{file_path}.csv"

        segments = predict(audio, model)
        postprocessor.write_csv(segments, file_path, csv_path)


def test_audio_with_transcript(
    audio_path: str,
    transcript_path: str,
    model: classifier.FrameClassifier,
    scaler: Any,
) -> None:
    """
    Test an audio file that has a corresponding transcription (as a JSON file)
    """
    log.info("Testing %s performance using transcript as ground-truth", model.name)

    audio = loader.load_audio(audio_path)
    transcript = dataset.load_test_transcription(transcript_path)

    segments = predict(audio, model, scaler)
    score = evaluator.foreground_overlap(
        transcript_segments=transcript, predicted_segments=segments
    )

    log.info("%s scored %.8f", model.name, score)


def main():
    configuration.load()

    model = train(
        model_name=configuration.get_required_str(configuration.MODEL_NAME),
        speech_dir=configuration.get_required_str(configuration.SPEECH_DIR),
        noise_dir=configuration.get_required_str(configuration.NOISE_DIR),
    )

    test_multiple_audio(configuration.get_required_str(configuration.TEST_DIR), model)


if __name__ == "__main__":
    main()
