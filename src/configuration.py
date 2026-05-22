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


SPEECH_DIR = "speech_dir"
NOISE_DIR = "noise_dir"
TEST_DIR = "test_dir"
RESULTS_DIR = "results_dir"
MLP_LAYER_SIZES = "layer_sizes"

MODEL_NAME = "model_name"
USE_CACHE = "use_cache"

REQUIRED_KEYS = set([SPEECH_DIR, NOISE_DIR, TEST_DIR, MODEL_NAME])

logging.addLevelName(logging.DEBUG, "DBG")
logging.addLevelName(logging.INFO, "INF")
logging.addLevelName(logging.WARNING, "WRN")
logging.addLevelName(logging.ERROR, "ERR")
logging.addLevelName(logging.CRITICAL, "CRT")


def _get_env(key: str) -> str | None:
    return os.getenv(f"{_ENV_PREFIX}_{key.upper()}")


def _get_required_env(key: str) -> str:
    value = _get_env(key.upper())

    if not value:
        raise ValueError(f"Required environment variable '{key}' has not been set!")

    return value


def _get_env_arguments() -> dict[str, Any | None]:
    return {
        SPEECH_DIR: _get_required_env(SPEECH_DIR),
        NOISE_DIR: _get_required_env(NOISE_DIR),
        TEST_DIR: _get_required_env(TEST_DIR),
        MODEL_NAME: _get_env(MODEL_NAME),
        RESULTS_DIR: _get_env(RESULTS_DIR),
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
        "-t",
        "--test",
        help="The test directory containing audio files which will be used for testing the classification process",
        type=str,
        required=False,
    )

    parser.add_argument(
        "-s",
        "--speech",
        help="The speech directory containing audio files which will be used to train the classifier",
        type=str,
        required=False,
    )

    parser.add_argument(
        "-n",
        "--noise",
        help="The noise directory containing audio files which will be used to train the classifier",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--results",
        help="The directory where the results (CSV, analytics) will be stored",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--cache",
        help="Whether to use the cache for persisting trained models. Using '--no-cache' disables it. By default it is enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--layers",
        help="The hidden layer sizes of the MLP classifier.",
        nargs="+",
        type=int,
        required=False,
    )

    args = parser.parse_args()
    dict = {
        MODEL_NAME: args.model,
        USE_CACHE: args.cache,
        TEST_DIR: args.test,
        SPEECH_DIR: args.speech,
        NOISE_DIR: args.noise,
        RESULTS_DIR: args.results,
        MLP_LAYER_SIZES: args.layers,
    }

    return {key: value for key, value in dict.items() if value is not None}


def load() -> None:
    global _CONFIG

    load_dotenv()
    log_level = _get_env("LOG_LEVEL")

    logging.basicConfig(
        level=_LOG_LEVEL_DICT.get(log_level.upper() if log_level else "", logging.INFO),
        format="%(asctime)s.%(msecs)03d | %(levelname)s | [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    log = logging.getLogger(__name__)

    cli_args = _get_cli_arguments()
    env_args = _get_env_arguments()

    args = env_args | cli_args  # CLI args will overwrite ENV args on conflict
    args = {key: value for key, value in args.items() if value is not None}

    missing_args = REQUIRED_KEYS - set(args.keys())
    if len(missing_args) > 0:
        raise ValueError(f"MISSING ARGUMENTS! {missing_args}")

    log.debug("Parsed arguments: %s", args)

    _CONFIG = args


def get_all() -> dict[str, Any]:
    global _CONFIG
    return _CONFIG


def get(key: str) -> Any | None:
    global _CONFIG

    if _CONFIG is None:
        raise ValueError("Tried to retrieve configuration before ``load``")

    return _CONFIG.get(key)


def get_required_str(key: str) -> str:
    val = get(key)

    if not val:
        raise ValueError(f"Configuration key not found: {key}")

    return str(val)
