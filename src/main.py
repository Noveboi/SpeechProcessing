import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import loader
import processor

if __name__ == "__main__":
    audio = loader.load_audio("test.wav")
    emphasized = processor.pre_emphasis(audio)

    wavfile.write("em.wav", emphasized.sample_rate, emphasized.waveform)
