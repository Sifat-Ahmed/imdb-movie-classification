# knn.py
import numpy as np
from typing import List, Union
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class KNN:
    """KNN implementation that expects pre-vectorized inputs."""

    def __init__(self, k: int = 5, distance_metric: str = 'cosine'):
        if distance_metric not in ('cosine', 'euclidean'):
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: List[int]):
        """Store precomputed training vectors and labels."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.X_train = X  # already vectorized
        self.y_train = np.array(y)
        self.classes_ = np.unique(self.y_train)
        if self.k < 1 or self.k > len(self.y_train):
            raise ValueError(f"k must be in [1, {len(self.y_train)}]")
        self.is_trained = True
        return self

    def _calculate_distances(self, X_test: np.ndarray) -> np.ndarray:
        """Calculate distances between test and training samples using sklearn pairwise funcs."""
        if self.distance_metric == 'cosine':
            similarities = cosine_similarity(X_test, self.X_train)
            return 1.0 - similarities
        elif self.distance_metric == 'euclidean':
            return euclidean_distances(X_test, self.X_train)

    def predict(self, X: np.ndarray) -> List[Union[str, int]]:
        """Predict labels for pre-vectorized test samples."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        distances = self._calculate_distances(X)
        n = distances.shape[0]
        preds: List[Union[str, int]] = []
        k = min(self.k, len(self.y_train))

        for i in range(n):
            # use argpartition for speed; order neighbors only within top-k if needed
            nn_idx = np.argpartition(distances[i], k - 1)[:k]
            nn_labels = self.y_train[nn_idx]
            preds.append(Counter(nn_labels).most_common(1)[0][0])

        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities via inverse-distance weighting."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        distances = self._calculate_distances(X)
        classes = self.classes_
        class_to_idx = {c: i for i, c in enumerate(classes)}
        n = distances.shape[0]
        k = min(self.k, len(self.y_train))
        probs = []

        for i in range(n):
            nn_idx = np.argpartition(distances[i], k - 1)[:k]
            nn_d = distances[i][nn_idx]
            nn_labels = self.y_train[nn_idx]

            # inverse distance weights (protect zero)
            w = 1.0 / (nn_d + 1e-8)
            totals = np.zeros(len(classes), dtype=float)
            for lab, ww in zip(nn_labels, w):
                totals[class_to_idx[lab]] += ww

            s = totals.sum()
            probs.append(totals / s if s > 0 else np.ones(len(classes)) / len(classes))

        return np.vstack(probs)
