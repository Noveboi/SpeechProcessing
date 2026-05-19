import logging
import os

from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

import classifier
import dataset
import extractor
import postprocessor
import preprocessor

ENV_PREFIX = "SPEECH"
LOG_LEVEL_DICT: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
}

logging.addLevelName(logging.DEBUG, "DBG")
logging.addLevelName(logging.INFO, "INF")
logging.addLevelName(logging.WARNING, "WRN")
logging.addLevelName(logging.ERROR, "ERR")
logging.addLevelName(logging.CRITICAL, "CRT")


def get_env(key: str) -> str | None:
    return os.getenv(f"{ENV_PREFIX}_{key}")


def get_required_env(key: str) -> str:
    value = get_env(key)

    if not value:
        raise ValueError(f"Required environment variable '{key}' has not been set!")

    return value


def log_level_from_str(level: str | None) -> int:
    return LOG_LEVEL_DICT.get(level.upper() if level else "", logging.INFO)


def main():
    load_dotenv()  # Assumes .env file is in this same directory as the script (or a higher up one)
    log_level = get_env("LOG_LEVEL")

    logging.basicConfig(
        level=log_level_from_str(log_level),
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    log = logging.getLogger(__name__)

    speech_dir = get_required_env("SPEECH_DIR")
    noise_dir = get_required_env("NOISE_DIR")
    test_dir = get_required_env("TEST_DIR")
    model_name = get_required_env("MODEL")

    # Training
    clf = classifier.get(model_name)

    X_train, y_train = dataset.build(speech_dir=speech_dir, noise_dir=noise_dir)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = clf.fit(X_train_scaled, y_train)

    # Inference
    for path, audio in dataset.load_test_audio(test_dir):
        log.info("Processing and predicting audio for '%s'", path)

        frames = preprocessor.process(audio)
        features = extractor.extract(frames)  # (N_frames, N_features)
        predictions = model.predict(
            scaler.transform(features)  # pyright: ignore[reportArgumentType]
        )  # (N_frames,)

        # Post-processin
        file_name = f"{path.name}"
        postprocessor.process(
            predictions,
            audio_filename=file_name,
            output_path=f"results/{model_name}_{file_name}.csv",
        )


if __name__ == "__main__":
    main()
