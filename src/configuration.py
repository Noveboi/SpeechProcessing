import argparse
import logging
import os
from typing import Any

from dotenv import load_dotenv

_ENV_PREFIX = "SPEECH"
_LOG_LEVEL_DICT: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
}
_CONFIG: dict[str, Any] | None = None


SPEECH_DIR = "SPEECH_DIR"
NOISE_DIR = "NOISE_DIR"
TEST_DIR = "TEST_DIR"
MODEL_NAME = "MODEL_NAME"
USE_CACHE = "USE_CACHE"

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


def _get_env_arguments() -> dict[str, Any | None]:
    return {
        SPEECH_DIR: _get_required_env(SPEECH_DIR),
        NOISE_DIR: _get_required_env(NOISE_DIR),
        TEST_DIR: _get_required_env(TEST_DIR),
        MODEL_NAME: _get_env(MODEL_NAME),
    }


def _get_cli_arguments() -> dict[str, Any | None]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        help="The classifier to use (KNN or MLP)",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    return {MODEL_NAME: args.model, USE_CACHE: args.cache}


def load() -> dict[str, Any]:
    load_dotenv()
    log_level = _get_env("LOG_LEVEL")

    logging.basicConfig(
        level=_LOG_LEVEL_DICT.get(log_level.upper() if log_level else "", logging.INFO),
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    log = logging.getLogger(__name__)

    env_args = _get_env_arguments()
    cli_args = _get_cli_arguments()

    args = env_args | cli_args  # CLI args will overwrite ENV args on conflict
    args = {key: value for key, value in args.items() if value is not None}

    missing_args = REQUIRED_KEYS - set(args.keys())
    if len(missing_args) > 0:
        raise ValueError(f"MISSING ARGUMENTS! {missing_args}")

    log.debug("Parsed arguments: %s", args)

    _CONFIG = args
    return _CONFIG


def get(key: str) -> Any | None:
    if _CONFIG is None:
        raise ValueError("Tried to retrieve configuration before ``load``")

    return _CONFIG.get(key, default=None)
