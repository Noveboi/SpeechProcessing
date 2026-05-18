import logging
import os

from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

import classifier
import dataset
import postprocessor
import processor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


ENV_PREFIX = "SPEECH"


def get_required_env(key: str) -> str:
    value = os.getenv(f"{ENV_PREFIX}_{key}")

    if not value:
        raise ValueError(f"Required environment variable '{key}' has not been set!")

    return value


def main():
    log = logging.getLogger(__name__)
    load_dotenv()  # Assumes .env file is in this same directory as the script (or a higher up one)

    speech_dir = get_required_env("SPEECH_DIR")
    noise_dir = get_required_env("NOISE_DIR")
    test_dir = get_required_env("TEST_DIR")

    # Training
    X_train, y_train = dataset.build(speech_dir=speech_dir, noise_dir=noise_dir)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = classifier.KNN(k=5).fit(X_train_scaled, y_train)

    # Inference
    for path, audio in dataset.load_test_audio(test_dir):
        log.info("Processing and predicting audio for '%s'", path)

        features = processor.process(audio)  # (T, # of features)
        features = scaler.transform(features)  # normalize
        predictions = model.predict(
            features  # pyright: ignore[reportArgumentType]
        )  # (T,) raw

        # Post-processin
        segments = postprocessor.process(
            predictions,
            audio_filename="mixed.wav",
            output_path="results/mixed.csv",
        )

        print(segments)


if __name__ == "__main__":
    main()
