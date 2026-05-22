import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

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
    model_name: str, scaler: TransformerMixin, speech_dir: str, noise_dir: str
) -> classifier.FrameClassifier:
    """
    Train the classifier on speech/noise data.

    Returns
    --------
    The trained classifier, ready to make predictions.
    """
    clf = classifier.get(model_name)

    cache_dir = Path("_cache")
    clf_file_path = cache_dir / f"{clf.name}.pkl"
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(clf_file_path, "rb") as f:
            log.info("Loading %s model into memory from '%s'", clf.name, clf_file_path)
            clf = pickle.load(f)
    except OSError:
        log.info("No persisted model found for %s", clf.name)

        data = dataset.build(
            speech_dir=speech_dir,
            noise_dir=noise_dir,
        )

        if not data:
            log.fatal("Training could not finish")
            sys.exit(1)

        X_train, y_train = data
        X_train_scaled = scaler.fit_transform(X_train)

        clf.fit(X_train_scaled, y_train)

        with open(clf_file_path, "wb") as f:
            log.info("Saving model data of %s into '%s'", clf.name, clf_file_path)
            pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)

    return clf


def predict(
    audio: Audio,
    model: classifier.FrameClassifier,
    scaler,
    frame_ms: int = 25,
    hop_ms: int = 10,
) -> list[Segment]:
    frames = preprocessor.process(audio, frame_ms=frame_ms, hop_ms=hop_ms)
    features: np.ndarray = extractor.extract(frames)  # (N_frames, N_features)
    features = scaler.transform(features)
    predictions = model.predict(features)  # (N_frames,)
    segments = postprocessor.process(predictions, hop_ms=hop_ms)

    return segments


def test_multiple_audio(
    test_dir: str, model: classifier.FrameClassifier, scaler: Any
) -> None:
    """
    Predict foreground/background intervals in audio files contained in the ``test_dir`` directory.

    Parameters
    --------
    test_dir : str
        The directory containing the WAV audio files
    model : FrameClassifier
        The classifier to be used in the predictions/classifications (e.g: ``KNN`` or ``MLP``)
    scaler : Any
        A scaler that normalizes the extracted features (e.g: ``StandardScaler``)

    Side Effects
    --------
    After the predictions/classifications, the results are stored in CSV files.
    """
    for path, audio in dataset.load_test_audio(test_dir):
        log.info("Processing and predicting audio for '%s'", path)

        file_path = path.name
        csv_path = f"results/{model.name}_{file_path}.csv"

        segments = predict(audio, model, scaler)
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
    config = configuration.get()

    model_name = config[configuration.MODEL_NAME]
    scaler = StandardScaler()
    model = train(
        model_name=model_name,
        scaler=scaler,
        speech_dir=config[configuration.SPEECH_DIR],
        noise_dir=config[configuration.NOISE_DIR],
    )

    test_multiple_audio(
        test_dir=config[configuration.TEST_DIR], model=model, scaler=scaler
    )


if __name__ == "__main__":
    main()
