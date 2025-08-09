import numpy as np
from scipy.sparse import issparse, csr_matrix

class NaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.is_trained = False

    def fit(self, X, y):
        # accept csr or dense
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes, dtype=np.float32)
        feat_counts = np.zeros((n_classes, n_features), dtype=np.float32)

        if issparse(X):
            X = X.tocsr()
            n_samples = X.shape[0]
            for idx, c in enumerate(self.classes_):
                rows = np.where(y == c)[0]
                # sum rows for this class; stays sparse then densify the 1×n_features result only
                class_sum = X[rows].sum(axis=0)
                feat_counts[idx, :] = np.asarray(class_sum).ravel().astype(np.float32)
                self.class_log_prior_[idx] = np.log(len(rows) / n_samples)
        else:
            n_samples = X.shape[0]
            for idx, c in enumerate(self.classes_):
                feat_counts[idx, :] = X[y == c].sum(axis=0).astype(np.float32)
                self.class_log_prior_[idx] = np.log((y == c).sum() / n_samples)

        smoothed = feat_counts + self.alpha
        self.feature_log_prob_ = (
            np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        ).astype(np.float32)

        self.is_trained = True
        return self

    def predict_log_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        if issparse(X):
            X = X.tocsr()
            jll = X @ self.feature_log_prob_.T  # sparse @ dense -> dense
        else:
            jll = X @ self.feature_log_prob_.T
        return (jll + self.class_log_prior_).astype(np.float32)

    def predict(self, X):
        return self.predict_log_proba(X).argmax(axis=1)