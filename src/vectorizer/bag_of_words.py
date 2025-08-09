import math
import numpy as np
from collections import Counter
from typing import List, Iterable, Tuple, Union

class BoWVectorizer:
    """
    Bag-of-Words vectorizer for pre-tokenized documents.
    - Input to fit/transform: List[List[str]]  (each doc is a list of tokens)
    - Output: np.ndarray of shape (n_docs, n_features)
    - binary=True -> presence/absence; binary=False -> counts
    - Supports min_df (int or float)
    """

    def __init__(self, binary: bool = True, min_df: Union[int, float] = 1):
        self.binary = bool(binary)
        self.min_df = min_df
        self.vocab_ = {}
        self.feature_names_ = []
        self.fitted_ = False

    def _passes_min_df(self, df_count: int, N: int) -> bool:
        if isinstance(self.min_df, float):
            return df_count >= self.min_df * N
        return df_count >= int(self.min_df)

    def fit(self, docs_tokens: List[List[str]]):
        N = len(docs_tokens)
        df = Counter()
        for toks in docs_tokens:
            df.update(set(toks))  # document frequency

        terms = [t for t, c in df.items() if self._passes_min_df(c, N)]
        terms.sort()  # deterministic
        self.vocab_ = {t: i for i, t in enumerate(terms)}
        self.feature_names_ = terms
        self.fitted_ = True
        return self

    def transform(self, docs_tokens: List[List[str]]) -> np.ndarray:
        if not self.fitted_:
            raise ValueError("Call fit before transform.")
        V = len(self.vocab_)
        X = np.zeros((len(docs_tokens), V), dtype=np.float32)
        for i, toks in enumerate(docs_tokens):
            if self.binary:
                for t in set(toks):
                    j = self.vocab_.get(t)
                    if j is not None:
                        X[i, j] = 1.0
            else:
                for t in toks:
                    j = self.vocab_.get(t)
                    if j is not None:
                        X[i, j] += 1.0
        return X

    def fit_transform(self, docs_tokens: List[List[str]]) -> np.ndarray:
        self.fit(docs_tokens)
        return self.transform(docs_tokens)

    def get_feature_names_out(self) -> List[str]:
        return list(self.feature_names_)