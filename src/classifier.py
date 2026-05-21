"""
This module contains the classification procedures for determining
background (noise) vs foreground (speech) in an audio signal.

It is given the feature matrix extracted from the ``processor.py``
and trains/tests from those features

NOTE:
    This code file has a lot of ignore statements for the linter I'm using (basedpyright, ruff).
    This is due to sklearn's types being very general, returning stuff like
    `float | Any` instead of `float | np.ndarray`. Since I'm using strict type annotations,
    I needed to cut some corners to have 0 errors!
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

log = logging.getLogger(__name__)


class FrameClassifier(ABC):
    """
    Abstract base class for frame-level classifiers.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The display name of the classifier."""
        ...

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FrameClassifier":
        """Train on feature matrix X and binary labels y."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 = noise, 1 = speech) for each frame."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(speech) in [0, 1] for each frame."""
        ...


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
            n_jobs=-1,  # use CPU all cores
        )

    @property
    def name(self) -> str:
        return "KNN"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        log.info("Training k-NN  (k=%d)  on %d frames", self.k, len(X))
        self._classifier.fit(X, y)
        log.info("k-NN training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        log.debug("k-NN predicting %d frames", len(X))
        predictions = self._classifier.predict(X)
        log.debug("k-NN predicted %d frames", len(X))
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Column 1 is P(speech=1)
        return self._classifier.predict_proba(X)[:, 1]  # pyright: ignore[reportArgumentType, reportCallIssue]


class MLP(FrameClassifier):
    """
    2-layer MLP classifier.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, int]
        Number of neurons in each hidden layer (default: (128, 64)).
    learning_rate : float
        Initial learning rate for Adam (default: 1e-3).
    max_iter : int
        Maximum number of training epochs (default: 200).
    """

    def __init__(
        self,
        layer_sizes: tuple = (256, 128, 64),
        learning_rate: float = 1e-3,
        max_iter: int = 500,
    ) -> None:
        self.hidden_layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self._classifier = MLPClassifier(
            hidden_layer_sizes=layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate,
            learning_rate="adaptive",  # reduces learning rate on plateau instead of stopping = more opportunity for improvement
            max_iter=max_iter,
            early_stopping=True,  # reserves 10% of the training data as
            # internal validation set and halts after
            # validation loss stops improving for
            # ``n_iter_no_change``.
            # Used for overfitting prevention
            validation_fraction=0.1,
            random_state=1337,  # constant seed for deterministic outputs!!
            n_iter_no_change=20,  # stop if val loss doesn't improve for 15 consecutive epochs
            verbose=True,  # keep logging enabled
        )

    @property
    def name(self) -> str:
        return "MLP"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        log.info(
            "Training MLP  (layers=%s, lr=%.0e, max_iter=%d)  on %d frames",
            self.hidden_layer_sizes,
            self.learning_rate,
            self.max_iter,
            len(X),
        )
        self._classifier.fit(X, y)
        log.info(
            "MLP training complete — stopped at epoch %d  |  best val accurary: %.4f",
            self._classifier.n_iter_,
            self._classifier.best_validation_score_,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._classifier.predict(X)  # pyright: ignore[reportReturnType]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._classifier.predict_proba(X)[:, 1]  # pyright: ignore[reportReturnType, reportArgumentType, reportCallIssue]


def get(key: str) -> FrameClassifier:
    """
    Get a frame classifier implemention based on a key string (case-insensitive).
    """
    clean_key = key.strip().upper().replace("-", "")

    if clean_key == "KNN":
        return KNN()
    if clean_key == "MLP":
        return MLP()

    raise ValueError(f"Unknown classifier type '{key}'")
