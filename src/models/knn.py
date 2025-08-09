import numpy as np
from scipy.sparse import issparse, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class KNN:
    def __init__(self, k=7, distance_metric="cosine", batch_size=2000):
        self.k = int(k)
        self.distance_metric = distance_metric
        self.batch_size = int(batch_size)
        self.is_trained = False

    def fit(self, X, y):
        # accept dense or sparse; keep as CSR if sparse
        self.X_train = X.tocsr() if issparse(X) else X
        self.y_train = np.asarray(y)
        n_samples = self.X_train.shape[0]   # <-- FIX: don't use len(X)
        if n_samples != len(self.y_train):
            raise ValueError("X and y must have the same number of rows")
        self.is_trained = True
        return self

    def _predict_batch(self, Xb):
        if self.distance_metric != "cosine":
            raise NotImplementedError("Only cosine is implemented sparsely.")
        # cosine similarity; keep result sparse
        sims = cosine_similarity(Xb, self.X_train, dense_output=False)
        preds = np.empty(Xb.shape[0], dtype=int)
        for i in range(Xb.shape[0]):
            row = sims.getrow(i)
            if row.nnz == 0:
                preds[i] = 0  # fallback to majority class or 0
                continue
            k = min(self.k, row.nnz)
            # top-k indices by similarity
            topk_idx = np.argpartition(row.data, -k)[-k:]
            nn_indices = row.indices[topk_idx]
            nn_labels = self.y_train[nn_indices]
            # majority vote
            vals, counts = np.unique(nn_labels, return_counts=True)
            preds[i] = vals[counts.argmax()]
        return preds

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        Xb = X.tocsr() if issparse(X) else X
        n = Xb.shape[0]
        out = np.empty(n, dtype=int)
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            out[start:stop] = self._predict_batch(Xb[start:stop])
        return out
