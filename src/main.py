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

    data = dataset.create(
        speech_dir=args.speech_files_path,
        noise_dir=args.noise_files_path,
    )

    if not data:
        log.fatal("Training could not finish")
        sys.exit(1)

    X_train, y_train = data

    clf.fit(X_train, y_train)
    cache.dump(clf, clf.cache_path)


def run_test_program(args: configuration.TestingConfiguration) -> None:
    log.info("Running testing program (%s)", args)
    test_audio = dataset.load_test_audio(args.test_files_path)

    log.info("%d test WAV files found in %s", len(test_audio), args.test_files_path)

    if len(test_audio) == 0:
        return

    clf = classifier.get(args.model_name, layers=args.layers)
    stored_clf = cache.load(clf.cache_path)

    if not stored_clf:
        raise ValueError(f"The classifier {clf} has not been trained.")

    clf = stored_clf

    for path, audio in test_audio:
        log.info("Processing and predicting audio for '%s'", path)

        file_path = path.name
        csv_path = f"results/{clf.name}_{path.stem}.csv"

        segments = predict(audio, clf)
        files.write_csv(segments, file_path, csv_path)


def run_evaluation_program(args: configuration.EvaluationConfiguration) -> None:
    log.info("Running evaluation program (%s)", args)

    transcription_segments = dataset.load_test_transcription(args.transcript_json_path)
    predicted_segments = files.load_csv_as_segments(args.csv_path)

    if not transcription_segments:
        raise ValueError("Transcription JSON path not found")

    if not predicted_segments:
        raise ValueError("CSV file not found")

    evaluation = evaluator.evaluate(
        prediction_segments=predicted_segments,
        ground_truth_segments=transcription_segments,
    )

    with open("evaluation.json", mode="w") as f:
        json.dump(dataclasses.asdict(evaluation), f, indent=2, sort_keys=True)
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
