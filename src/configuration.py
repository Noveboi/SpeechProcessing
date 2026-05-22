import argparse
import logging
import os

from dotenv import load_dotenv

_ENV_PREFIX = "SPEECH"
_LOG_LEVEL_DICT: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
}

SPEECH_DIR = "SPEECH_DIR"
NOISE_DIR = "NOISE_DIR"
TEST_DIR = "TEST_DIR"
MODEL_NAME = "MODEL"

REQUIRED_KEYS = set([SPEECH_DIR, NOISE_DIR, TEST_DIR, MODEL_NAME])

logging.addLevelName(logging.DEBUG, "DBG")
logging.addLevelName(logging.INFO, "INF")
logging.addLevelName(logging.WARNING, "WRN")
logging.addLevelName(logging.ERROR, "ERR")
logging.addLevelName(logging.CRITICAL, "CRT")


def _get_env(key: str) -> str | None:
    return os.getenv(f"{_ENV_PREFIX}_{key}")


def _get_required_env(key: str) -> str:
    value = _get_env(key)

    if not value:
        raise ValueError(f"Required environment variable '{key}' has not been set!")

    return value


def _get_env_arguments() -> dict[str, str | None]:
    return {
        SPEECH_DIR: _get_required_env(SPEECH_DIR),
        NOISE_DIR: _get_required_env(NOISE_DIR),
        TEST_DIR: _get_required_env(TEST_DIR),
        MODEL_NAME: _get_env(MODEL_NAME),
    }


def _get_cli_arguments() -> dict[str, str | None]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        help="The classifier to use (KNN or MLP)",
        type=str,
        required=False,
    )

    args = parser.parse_args()

    return {MODEL_NAME: args.model}


def get() -> dict[str, str]:
    load_dotenv()
    log_level = _get_env("LOG_LEVEL")

    logging.basicConfig(
        level=_LOG_LEVEL_DICT.get(log_level.upper() if log_level else "", logging.INFO),
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    env_args = _get_env_arguments()
    cli_args = _get_cli_arguments()

    args = env_args | cli_args  # CLI args will overwrite ENV args on conflict
    args = {key: value for key, value in args.items() if value is not None}

    missing_args = REQUIRED_KEYS - set(args.keys())
    if len(missing_args) > 0:
        raise ValueError(f"MISSING ARGUMENTS! {missing_args}")

    return args
