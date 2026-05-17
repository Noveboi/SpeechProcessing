"""
This module contains the classification procedures for determining background (noise) vs foreground (speech) in an audio signal.

It is given the feature matrix extracted from the ``processor.py`` and trains/tests from those features
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


class FrameClassifier(ABC):
    """
    Abstract base class for frame-level classifiers.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FrameClassifier":
        """Train on feature matrix X and binary labels y."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 = noise, 1 = speech) for each frame."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(speech) in [0, 1] for each frame."""


class KNN(FrameClassifier):
    """
    k-Nearest Neighbours classifier.

    Parameters
    ----------
    k : int
        Number of neighbours (default: 5).
        IMPORTANT: Use an odd number to avoid ties.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._classifier = KNeighborsClassifier(
            n_neighbors=k,
            metric="euclidean",  # this is simple and the best choice because the data is normalized
            n_jobs=-1,  # use all cores
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        logger.info("Training k-NN  (k=%d)  on %d frames", self.k, len(X))
        self._classifier.fit(X, y)
        logger.info("k-NN training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = self._classifier.predict(X)
        logger.debug("k-NN predicted %d frames", len(X))
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Column 1 is P(speech=1)
        return self._classifier.predict_proba(X)[:, 1]  # pyright: ignore[reportArgumentType, reportCallIssue]
