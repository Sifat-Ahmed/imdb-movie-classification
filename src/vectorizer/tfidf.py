import math
from collections import Counter
from typing import List

import numpy as np
from scipy.sparse import csr_matrix


class TfidfVectorizerScratch:
    """
    TF-IDF for pre-tokenized docs.
    Input:  List[List[str]]
    Output: scipy.sparse.csr_matrix (n_docs, n_features)

    idf = log((1 + N) / (1 + df)) + 1
    sublinear_tf: tf' = 1 + log(tf)
    l2_norm: per-document L2 normalization
    """

    def __init__(self, min_df=1, sublinear_tf: bool = True, l2_norm: bool = True, verbose: bool = False):
        self.min_df = min_df
        self.sublinear_tf = bool(sublinear_tf)
        self.l2_norm = bool(l2_norm)
        self.verbose = bool(verbose)

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
        df = Counter()
        for toks in docs_tokens:
            if toks:
                df.update(set(toks))

        terms = [t for t, c in df.items() if self._passes_min_df(c, N)]
        terms.sort()
        self.vocab_ = {t: i for i, t in enumerate(terms)}
        self.feature_names_ = terms

        V = len(self.vocab_)
        idf = np.zeros(V, dtype=np.float32)
        for t, j in self.vocab_.items():
            dft = df[t]
            idf[j] = math.log((1 + N) / (1 + dft)) + 1.0
        self.idf_ = idf
        self.fitted_ = True

        if self.verbose:
            mf = f"{self.min_df}" if not isinstance(self.min_df, float) else f"{self.min_df:.3f}"
            print(f"[TFIDF] fit: docs={N}, vocab_size={V}, min_df={mf}")
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

            counts = Counter()
            for t in toks:
                j = self.vocab_.get(t)
                if j is not None:
                    counts[j] += 1

            if not counts:
                continue

            values = {}
            for j, cnt in counts.items():
                tf = (1.0 + math.log(cnt)) if self.sublinear_tf else float(cnt)
                val = tf * float(self.idf_[j])
                values[j] = val

            if self.l2_norm:
                norm = math.sqrt(sum(v * v for v in values.values()))
                if norm > 0:
                    inv = 1.0 / norm
                    for j in values:
                        values[j] *= inv

            for j, v in values.items():
                rows.append(i)
                cols.append(j)
                data.append(np.float32(v))

        X = csr_matrix(
            (np.array(data, dtype=np.float32),
             (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
            shape=(N, V),
            dtype=np.float32,
        )

        if self.verbose:
            nnz = X.nnz
            density = nnz / (N * V) if N and V else 0.0
            print(f"[TFIDF] transform: shape={X.shape}, nnz={nnz}, density={density:.6f}")
        return X

    def fit_transform(self, docs_tokens: List[List[str]]):
        return self.fit(docs_tokens).transform(docs_tokens)

    def get_feature_names_out(self) -> List[str]:
        return list(self.feature_names_)