import math
import numpy as np
from collections import Counter
from typing import List, Iterable, Tuple, Union


class TfidfVectorizerScratch:
    """
    TF-IDF vectorizer for pre-tokenized documents.
    - Input: List[List[str]]
    - Output: np.ndarray (n_docs, n_features)
    - idf = log((1 + N) / (1 + df)) + 1
    - sublinear_tf: tf' = 1 + log(tf) if tf>0 else 0
    - l2 normalize by default
    - Supports min_df (int or float)
    """

    def __init__(
        self,
        min_df: Union[int, float] = 1,
        sublinear_tf: bool = True,
        l2_norm: bool = True
    ):
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.l2_norm = l2_norm

        self.vocab_ = {}
        self.feature_names_ = []
        self.idf_ = None
        self.fitted_ = False

    def _passes_min_df(self, df_count: int, N: int) -> bool:
        if isinstance(self.min_df, float):
            return df_count >= self.min_df * N
        return df_count >= int(self.min_df)

    def fit(self, docs_tokens: List[List[str]]):
        N = len(docs_tokens)
        # document frequency
        df = Counter()
        for toks in docs_tokens:
            df.update(set(toks))

        terms = [t for t, c in df.items() if self._passes_min_df(c, N)]
        terms.sort()  # deterministic vocab order
        self.vocab_ = {t: i for i, t in enumerate(terms)}
        self.feature_names_ = terms

        # idf
        V = len(self.vocab_)
        idf = np.zeros(V, dtype=np.float32)
        for t, j in self.vocab_.items():
            dft = df[t]
            idf[j] = math.log((1 + N) / (1 + dft)) + 1.0
        self.idf_ = idf
        self.fitted_ = True
        return self

    def transform(self, docs_tokens: List[List[str]]) -> np.ndarray:
        if not self.fitted_:
            raise ValueError("Call fit before transform.")
        V = len(self.vocab_)
        X = np.zeros((len(docs_tokens), V), dtype=np.float32)

        for i, toks in enumerate(docs_tokens):
            # term frequency per doc
            tf = Counter()
            for t in toks:
                j = self.vocab_.get(t)
                if j is not None:
                    tf[j] += 1

            if self.sublinear_tf:
                for j, cnt in tf.items():
                    X[i, j] = (1.0 + math.log(cnt)) * self.idf_[j]
            else:
                for j, cnt in tf.items():
                    X[i, j] = float(cnt) * self.idf_[j]

            if self.l2_norm:
                n = np.linalg.norm(X[i])
                if n > 0:
                    X[i] /= n

        return X

    def fit_transform(self, docs_tokens: List[List[str]]) -> np.ndarray:
        self.fit(docs_tokens)
        return self.transform(docs_tokens)

    def get_feature_names_out(self) -> List[str]:
        return list(self.feature_names_)