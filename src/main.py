import logging
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

import classifier
import configuration
import dataset
import extractor
import postprocessor
import preprocessor
from src.common import Audio, Segment

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

    X_train, y_train = dataset.build(
        speech_dir=speech_dir,
        noise_dir=noise_dir,
    )

    X_train_scaled = scaler.fit_transform(X_train)

    model = clf.fit(X_train_scaled, y_train)

    return model


def predict(audio: Audio, model: classifier.FrameClassifier, scaler) -> list[Segment]:
    frames = preprocessor.process(audio)
    features: np.ndarray = extractor.extract(frames)  # (N_frames, N_features)
    features = scaler.transform(features)  # pyright: ignore[reportAssignmentType]
    predictions = model.predict(features)  # (N_frames,)
    segments = postprocessor.process(predictions)

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
    """
    for path, audio in dataset.load_test_audio(test_dir):
        log.info("Processing and predicting audio for '%s'", path)

        file_path = path.name
        csv_path = f"results/{model.name}_{file_path}.csv"

        segments = predict(audio, model, scaler)
        postprocessor.write_csv(segments, file_path, csv_path)


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


if __name__ == "__main__":
    main()
