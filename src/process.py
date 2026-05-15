import numpy as np


class AudioProcessor:
    def __init__(self, audio: np.ndarray, sample_rate: int):
        self._audio = audio
        self._fs = sample_rate

        self._channels: int = audio.ndim
        self._length: int = audio.shape[0] / self._fs

    def _print(self):
        """
        DEBUG METHOD!
        """
        print(
            f"{self._fs} Hz, {self._channels} channel(s), {self._length} secs, {self._audio.shape}"
        )
