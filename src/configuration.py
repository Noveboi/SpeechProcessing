import argparse
import logging
import os
from abc import ABC
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv

import classifier


class SubprogramConfiguration(ABC): ...


@dataclass(frozen=True)
class TrainingConfiguration(SubprogramConfiguration):
    model_name: str
    speech_files_path: str
    noise_files_path: str
    layers: tuple[int, ...] | None


@dataclass(frozen=True)
class TestingConfiguration(SubprogramConfiguration):
    model_name: str
    test_files_path: str
    results_directory: str
    layers: tuple[int, ...] | None


@dataclass(frozen=True)
class EvaluationConfiguration(SubprogramConfiguration):
    transcript_json_path: str
    results_path: str
    csv_path: str


_ENV_PREFIX = "SPEECH"
_LOG_LEVEL_DICT: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
}

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


def cli(
    train_program: Callable[[TrainingConfiguration], None],
    test_program: Callable[[TestingConfiguration], None],
    evaluation_program: Callable[[EvaluationConfiguration], None],
) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_choices = classifier.CLASSIFIERS.keys()

    train_parser = subparsers.add_parser(
        name="train",
        help="Train the classifier on speech and noise data.",
    )

    test_parser = subparsers.add_parser(
        name="test",
        help="Test the trained classifier.",
    )

    evaluate_parser = subparsers.add_parser(
        name="evaluate",
        help="Evaluate the tests using ground-truth transcripts.",
    )

    train_parser.add_argument(
        "-m",
        "--model",
        help="The classifier to use",
        choices=model_choices,
        required=True,
    )

    train_parser.add_argument(
        "-s",
        "--speech",
        help="The speech directory containing audio files which will be used to train the classifier",
        type=str,
        required=True,
    )

    train_parser.add_argument(
        "-n",
        "--noise",
        help="The noise directory containing audio files which will be used to train the classifier",
        type=str,
        required=True,
    )

    train_parser.add_argument(
        "--layers",
        help="The hidden layer sizes of the MLP classifier.",
        nargs="+",
        type=int,
        required=False,
    )

    test_parser.add_argument(
        "--results",
        help="The directory where the results (CSV, analytics) will be stored",
        type=str,
        required=True,
    )

    test_parser.add_argument(
        "-m",
        "--model",
        help="The classifier to use",
        choices=model_choices,
        required=True,
    )

    test_parser.add_argument(
        "-t",
        "--test",
        help="The test directory containing audio files which will be used for testing the classification process",
        type=str,
        required=True,
    )

    test_parser.add_argument(
        "--layers",
        help="The hidden layer sizes of the MLP classifier.",
        nargs="+",
        type=int,
        required=False,
    )

    evaluate_parser.add_argument(
        "-t",
        "--transcript",
        help="The path to the JSON transcription",
        type=str,
        required=True,
    )

    evaluate_parser.add_argument(
        "-c",
        "--csv",
        help="The path to the predictions in CSV format",
        type=str,
        required=True,
    )

    evaluate_parser.add_argument(
        "-r",
        "--results",
        help="The path to the results folder where the evaluation will be stored",
        type=str,
        required=True,
    )

    train_parser.set_defaults(func=_config_train_subprogram(train_program))
    test_parser.set_defaults(func=_config_test_subprogram(test_program))
    evaluate_parser.set_defaults(func=_config_evaluate_subprogram(evaluation_program))

    args = parser.parse_args()
    args.func(args)


def _config_train_subprogram(func: Callable[[TrainingConfiguration], None]):
    return lambda args: func(
        TrainingConfiguration(
            model_name=args.model,
            layers=args.layers,
            speech_files_path=args.speech,
            noise_files_path=args.noise,
        )
    )


def _config_test_subprogram(func: Callable[[TestingConfiguration], None]):
    return lambda args: func(
        TestingConfiguration(
            model_name=args.model,
            layers=args.layers,
            test_files_path=args.test,
            results_directory=args.results,
        )
    )


def _config_evaluate_subprogram(func: Callable[[EvaluationConfiguration], None]):
    return lambda args: func(
        EvaluationConfiguration(
            transcript_json_path=args.transcript,
            csv_path=args.csv,
            results_path=args.results,
        )
    )


def load() -> None:
    load_dotenv()
    log_level = _get_env("LOG_LEVEL")

    logging.basicConfig(
        level=_LOG_LEVEL_DICT.get(log_level.upper() if log_level else "", logging.INFO),
        format="%(asctime)s.%(msecs)03d | %(levelname)s | [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
