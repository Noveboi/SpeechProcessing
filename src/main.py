import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import loader
import plots
import processor


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    audio = loader.load_audio("test.wav")
    audio_emphasis = processor.pre_emphasis(audio)
    frames = processor.split_into_frames(audio_emphasis, 25, 10, "hamming")
    features = processor.extract(frames)
    mfcc_matrix = features[:, 2:41]

    plots.plot_waveform(audio, audio_emphasis)
    plots.plot_power_spectrum(frames[50])
    plots.plot_mel_filterbank(frames[0])
    plots.plot_spectrogram(frames)
    plots.plot_mfccs(mfcc_matrix)
    plots.plot_time_domain_features(features)


if __name__ == "__main__":
    main()
