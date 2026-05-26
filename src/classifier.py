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

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


class FrameClassifier(ABC):
    """
    Abstract base class for frame-level classifiers.
    """

    @property
    def cache_path(self) -> str:
        return f"{self.name}_{self.hash()}.pkl"

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
    def hash(self) -> str:
        """Hash based on attributes"""
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

    def __init__(self, k: int = 5, **kwargs) -> None:
        self.k = k

        classifier = KNeighborsClassifier(
            n_neighbors=k,
            algorithm="kd_tree",
            n_jobs=-1,  # use CPU all cores
        )

        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", classifier),
            ]
        )

    @property
    def name(self) -> str:
        return "KNN"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        log.info("Training k-NN  (k=%d)  on %d frames", self.k, len(X))
        self._pipeline.fit(X, y)
        log.info("k-NN training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        log.debug("k-NN predicting %d frames", len(X))
        predictions = self._pipeline.predict(X)
        log.debug("k-NN predicted %d frames", len(X))
        return predictions

    def hash(self) -> str:
        payload = {
            "classifier": self.name,
            "k": self.k,
        }

        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def __str__(self) -> str:
        return f"KNN(k={self.k})"


class MLP(FrameClassifier):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        max_iter: int = 500,
        **kwargs,
    ) -> None:
        self.hidden_layer_sizes = tuple(kwargs.get("layers") or (64, 32))
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        classifier = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=1337,  # constant seed for deterministic fitting!!
            n_iter_no_change=20,
            verbose=True,
        )

        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", classifier),
            ]
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
        self._pipeline.fit(X, y)
        log.info("MLP training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)  # pyright: ignore[reportReturnType]

    def hash(self) -> str:
        payload = {
            "classifier": self.name,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
        }

        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def __str__(self) -> str:
        return f"MLP({self.hidden_layer_sizes}, {self.learning_rate}, {self.max_iter})"


CLASSIFIERS: dict[str, Callable[[dict[str, Any]], FrameClassifier]] = {
    "KNN": lambda kwargs: KNN(**kwargs),
    "MLP": lambda kwargs: MLP(**kwargs),
}


def get(key: str, **kwargs) -> FrameClassifier:
    """
    Get a frame classifier implemention based on a key string (case-insensitive).
    """
    clean_key = key.strip().upper().replace("-", "")

    classifier_factory = CLASSIFIERS.get(clean_key)

    if not classifier_factory:
        raise ValueError(
            f"Unknown classifier type '{key}'. Available: {CLASSIFIERS.keys()}"
        )

    return classifier_factory(kwargs)
