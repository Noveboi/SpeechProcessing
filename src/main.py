import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import loader
import processor


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    audio = loader.load_audio("test.wav")
    emphasized = processor.pre_emphasis(audio)

    wavfile.write("em.wav", emphasized.sample_rate, emphasized.waveform)
