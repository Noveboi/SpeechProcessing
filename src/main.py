from scipy.io import wavfile

from process import AudioProcessor

if __name__ == "__main__":
    fs, audio = wavfile.read("test.wav")
    processor = AudioProcessor(audio, fs)

    processor._print()
