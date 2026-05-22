#!/usr/bin/python3

import logging
import sys
from pathlib import Path

import numpy as np

import cache
import classifier
import configuration
import dataset
import extractor
import files
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
    clf = classifier.get(model_name, **configuration.get_all())
    clf_file_path = f"{clf.name}_{clf.hash()}.pkl"

    stored_clf = cache.load(clf_file_path)
    if stored_clf:
        return stored_clf

    log.info("No persisted model found for %s at '%s'", clf.name, clf_file_path)

    data = dataset.create(
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
    """
    Use the train classifier to make predictions about some ``audio``.

    This invokes the full pipeline, from pre-processing to post-processing.
    """
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
    test_audio = dataset.load_test_audio(test_dir)

    log.info("%d test WAV files found in %s", len(test_audio), test_dir)

    for path, audio in test_audio:
        log.info("Processing and predicting audio for '%s'", path)

        file_path = path.name
        csv_path = f"results/{model.name}_{Path(file_path).stem}.csv"

        segments = predict(audio, model)
        files.write_csv(segments, file_path, csv_path)


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
