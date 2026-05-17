import logging

from sklearn.preprocessing import StandardScaler

import classifier
import dataset
import loader
import postprocessor
import processor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    # Training
    X_train, y_train = dataset.build(
        speech_dir="../samples/train/speech", noise_dir="../samples/train/noise"
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = classifier.KNN(k=5).fit(X_train_scaled, y_train)

    # Inference
    audio = loader.load_audio("test.wav")
    features = processor.process(audio)  # (T, 45)
    features = scaler.transform(features)  # normalise
    predictions = model.predict(
        features  # pyright: ignore[reportArgumentType]
    )  # (T,) raw

    # Post-processin
    segments = postprocessor.process(
        predictions,
        audio_filename="mixed.wav",
        output_path="results/mixed.csv",
    )


if __name__ == "__main__":
    main()
