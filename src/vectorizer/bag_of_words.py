import math
import time
from collections import Counter
from typing import List

import numpy as np
from scipy.sparse import csr_matrix


class BoWVectorizer:
    """
    Bag-of-Words vectorizer for pre-tokenized documents.
    Input: List[List[str]]  (each doc is a list of tokens)
    Output: scipy.sparse.csr_matrix (n_docs, n_features)
    """

    def __init__(self, binary: bool = True, min_df=1, verbose: bool = False):
        self.binary = bool(binary)
        self.min_df = min_df
        self.verbose = bool(verbose)
        self.vocab_ = {}
        self.feature_names_ = []
        self.fitted_ = False

    def _passes_min_df(self, df_count: int, N: int) -> bool:
        if isinstance(self.min_df, float):
            return df_count >= self.min_df * N
        return df_count >= int(self.min_df)

    def fit(self, docs_tokens: List[List[str]]):
        t0 = time.perf_counter()
        N = len(docs_tokens)
        df = Counter()

        for toks in docs_tokens:
            if toks:
                df.update(set(toks))

        terms = [t for t, c in df.items() if self._passes_min_df(c, N)]
        terms.sort()
        self.vocab_ = {t: i for i, t in enumerate(terms)}
        self.feature_names_ = terms
        self.fitted_ = True

        if self.verbose:
            dt = time.perf_counter() - t0
            print(f"[BoWVectorizer] fit: docs={N}, vocab_size={len(terms)}, time={dt:.3f}s")
        return self

    def transform(self, docs_tokens: List[List[str]]):
        if not self.fitted_:
            raise ValueError("Call fit before transform.")
        V = len(self.vocab_)
        N = len(docs_tokens)

        rows, cols, data = [], [], []

        for i, toks in enumerate(docs_tokens):
            if not toks:
                continue
            if self.binary:
                for t in set(toks):
                    j = self.vocab_.get(t)
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
                        data.append(1.0)
            else:
                counts = Counter(toks)
                for t, c in counts.items():
                    j = self.vocab_.get(t)
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
                        data.append(float(c))

        X = csr_matrix((np.array(data, dtype=np.float32),
                        (np.array(rows, dtype=np.int32),
                         np.array(cols, dtype=np.int32))),
                       shape=(N, V), dtype=np.float32)

        if self.verbose:
            nnz = X.nnz
            density = nnz / (N * V) if N and V else 0
            print(f"[BoWVectorizer] transform: shape={X.shape}, nnz={nnz}, density={density:.6f}")
        return X

    def fit_transform(self, docs_tokens: List[List[str]]):
        return self.fit(docs_tokens).transform(docs_tokens)

    def get_feature_names_out(self) -> List[str]:
        return list(self.feature_names_)
