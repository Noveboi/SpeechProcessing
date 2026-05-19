"""
This module contains the classification procedures for determining
background (noise) vs foreground (speech) in an audio signal.

It is given the feature matrix extracted from the ``processor.py``
and trains/tests from those features

NOTE:
    This code file has a lot of ignore statements for the linter I'm using (basedpyright).
    This is due to sklearn's types being very general, returning stuff like
    `float | Any` instead of `float | np.ndarray`. Since I'm using strict type annotations,
    I needed to cut some corners to have 0 errors!
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

log = logging.getLogger(__name__)


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
            n_jobs=-1,  # use CPU all cores
        )

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
        layer_sizes: tuple[int, int] = (128, 64),
        learning_rate: float = 1e-3,
        max_iter: int = 200,
    ) -> None:
        self.hidden_layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self._classifier = MLPClassifier(
            hidden_layer_sizes=layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=True,  # reserves 10% of the training data as
            # internal validation set and halts after
            # validation loss stops improving for
            # ``n_iter_no_change``.
            # Used for overfitting prevention
            validation_fraction=0.1,
            random_state=1337,  # constant seed for deterministic outputs!!
            n_iter_no_change=15,  # stop if val loss doesn't improve for 15 consecutive epochs
            verbose=False,  # use our own loggig
        )

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
            "MLP training complete — stopped at epoch %d  |  best val loss: %.4f",
            self._classifier.n_iter_,
            self._classifier.best_validation_score_,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._classifier.predict(X)  # pyright: ignore[reportReturnType]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._classifier.predict_proba(X)[:, 1]  # pyright: ignore[reportReturnType, reportArgumentType, reportCallIssue]


def evaluate(
    classifier: FrameClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Evaluate a trained classifier and return a metrics dictionary.

    Computes accuracy, per-class precision / recall / F1, and a
    confusion matrix.

    Parameters
    ----------
    classifier : FrameClassifier
        The classifier which will be evaluated
    X : np.ndarray, shape (N, D)
        Normalised feature matrix
    y : np.ndarray, shape (N,)
        Ground-truth labels

    Returns
    -------
    metrics : dict with keys:
        accuracy, precision, recall, f1  (per-class arrays, index 0=noise/1=speech)
        confusion_matrix                 (2×2 np.ndarray)
    """
    y_pred = classifier.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, labels=[0, 1])  # pyright: ignore[reportAssignmentType]
    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray

    log.info("Accuracy  : %.4f", accuracy)
    log.info("           Noise      Speech")
    log.info("Precision : %.4f     %.4f", precision[0], precision[1])
    log.info("Recall    : %.4f     %.4f", recall[0], recall[1])
    log.info("F1        : %.4f     %.4f", f1[0], f1[1])
    log.info("Confusion matrix (rows=true, cols=pred):")
    log.info("           Pred noise  Pred speech")
    log.info("True noise     %6d       %6d", cm[0, 0], cm[0, 1])
    log.info("True speech    %6d       %6d", cm[1, 0], cm[1, 1])

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
