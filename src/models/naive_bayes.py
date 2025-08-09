import math
import numpy as np
from typing import List

class NaiveBayes:
    """Multinomial Naive Bayes for pre-vectorized (dense) inputs with integer labels."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.is_trained = False

    def fit(self, X: np.ndarray, y: List[int]):
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a NumPy array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        y = np.array(y, dtype=int)

        # Class priors
        class_counts = np.bincount(y, minlength=n_classes)
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())

        # Feature counts per class
        feat_counts = np.zeros((n_classes, n_features), dtype=float)
        for ci in range(n_classes):
            feat_counts[ci] = X[y == ci].sum(axis=0)

        # Likelihoods with Laplace smoothing
        alpha = self.alpha
        denom = feat_counts.sum(axis=1) + alpha * n_features
        self.feature_log_prob_ = np.log((feat_counts + alpha) / denom[:, None])

        self.is_trained = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Call fit before predict_proba")
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a NumPy array")

        logj = self.class_log_prior_ + X.dot(self.feature_log_prob_.T)
        max_logj = logj.max(axis=1, keepdims=True)
        probs = np.exp(logj - max_logj)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> List[int]:
        probs = self.predict_proba(X)
        return probs.argmax(axis=1).tolist()
