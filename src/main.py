#!/usr/bin/python3

import dataclasses
import json
import logging
import sys

import numpy as np

import cache
import classifier
import configuration
import dataset
import evaluator
import extractor
import files
import postprocessor
import preprocessor
from common import Audio, Segment

log = logging.getLogger(__name__)


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
    log.info("Pre-processing audio")
    frames = preprocessor.process(audio, frame_ms=frame_ms, hop_ms=hop_ms)

    log.info("Extracting features for %d frames", len(frames))
    features: np.ndarray = extractor.extract(frames)  # (N_frames, N_features)

    log.info("Predicting background/foreground for %d frames", len(frames))
    predictions = model.predict(features)  # (N_frames,)

    log.info("Post-processing %d segments", len(predictions))
    segments = postprocessor.process(predictions, hop_ms=hop_ms)

    return segments


def run_train_program(args: configuration.TrainingConfiguration) -> None:
    log.info("Running training program (%s)", args)
    clf = classifier.get(args.model_name, layers=args.layers)

    speech_dir = files.to_existing_path(args.speech_files_path)
    noise_dir = files.to_existing_path(args.noise_files_path)

    data = dataset.create(
        speech_dir=speech_dir,
        noise_dir=noise_dir,
    )

    if not data:
        log.fatal("Training could not finish")
        sys.exit(1)

    X_train, y_train = data

    clf.fit(X_train, y_train)
    cache.dump(clf, clf.cache_path)


def run_test_program(args: configuration.TestingConfiguration) -> None:
    log.info("Running testing program (%s)", args)
    test_dir = files.to_existing_path(args.test_files_path)
    test_audio = dataset.load_test_audio(test_dir)

    log.info("%d test WAV files found in %s", len(test_audio), args.test_files_path)

    if len(test_audio) == 0:
        return

    clf = classifier.get(args.model_name, layers=args.layers)
    stored_clf: classifier.FrameClassifier | None = cache.load(clf.cache_path)

    if not stored_clf:
        raise ValueError(f"The classifier {clf} has not been trained.")

    clf = stored_clf

    for path, audio in test_audio:
        log.info("Processing and predicting audio for '%s'", path)

        results_dir = files.to_existing_path(args.results_directory)
        csv_path = results_dir / f"{clf.name}_{clf.hash()}_{path.stem}.csv"

        segments = predict(audio, clf)
        files.write_csv(segments, path.name, csv_path)


def run_evaluation_program(args: configuration.EvaluationConfiguration) -> None:
    log.info("Running evaluation program (%s)", args)

    transcription_dir = files.to_existing_path(args.transcript_json_path)
    csv_dir = files.to_existing_path(args.csv_path)
    results_dir = files.to_existing_path(args.results_path)

    transcription_segments = dataset.load_transcription(transcription_dir)
    predicted_segments = files.load_csv_as_segments(csv_dir)

    if not transcription_segments:
        raise ValueError("Transcription JSON path not found")

    if not predicted_segments:
        raise ValueError("CSV file not found")

    evaluation = evaluator.evaluate(
        prediction_segments=predicted_segments,
        ground_truth_segments=transcription_segments,
    )

    evaluation_path = results_dir / "evaluation.json"

    with open(evaluation_path, mode="w") as f:
        json.dump(dataclasses.asdict(evaluation), f, indent=2)
        log.info("Evaluation saved at %s", f.name)


def main():
    configuration.load()

    configuration.cli(
        train_program=run_train_program,
        test_program=run_test_program,
        evaluation_program=run_evaluation_program,
    )


if __name__ == "__main__":
    main()
