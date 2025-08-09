# Vectorizers Module Documentation

This document describes the two custom vectorizer implementations — 
```
├── src/
│   ├── preprocessor/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── pos_tag.py
│   │   └── preprocessing.py
``` 

Both accept **pre-tokenized** text (list of tokens per document) and output a sparse matrix representation for machine learning models.

---

## 1. Bag-of-Words Vectorizer (`BoWVectorizer`)

### Purpose
Represents text purely based on **token counts** (or presence/absence if binary mode is enabled), ignoring word order.

### Key Parameters
- **binary**: If `True`, stores `1` for presence of a token; if `False`, stores raw counts.
- **min_df**: Minimum number (or proportion) of documents in which a term must appear to be kept.
- **verbose**: Prints diagnostic info if enabled.

### Workflow
1. **fit**: Builds a vocabulary mapping each term to a column index, filtering terms by `min_df`.
2. **transform**: Creates a sparse matrix with shape `(n_documents, vocab_size)` containing counts or binary indicators.
3. **fit_transform**: Combines both steps.

### Example
```python
from vectorizer.bag_of_words import BoWVectorizer

docs = [["i", "love", "this", "movie"],
        ["i", "hate", "this", "movie"]]

vec = BoWVectorizer(binary=False)
X = vec.fit_transform(docs)

print(vec.get_feature_names_out())  # ['hate', 'i', 'love', 'movie', 'this']
print(X.toarray())
```

Vocabulary
```
hate=0, i=1, love=2, movie=3, this=4
```
Vector
```
Doc 1: [0, 1, 1, 1, 1]   # no 'hate', one 'i', one 'love', etc.
Doc 2: [1, 1, 0, 1, 1]   # has 'hate', no 'love'
```


## 2. TF-IDF Vectorizer (`TfidfVectorizerScratch`)

### Purpose
Represents text by Term Frequency – Inverse Document Frequency scores, which balance term importance in a document against its frequency across all documents.

### Key Parameters
- min_df: Minimum number (or proportion) of documents in which a term must appear.
- sublinear_tf: If `True`, applies `tf' = 1 + log(tf)` to dampen term frequency growth.
- l2_norm: If `True`, normalizes each document vector to unit length.
- verbose: Prints diagnostic info if enabled.

### Formula
For term ***t*** in document ***d***:
- Term Frequency (TF):
```
tf = count(t, d)
```
or
```
tf' = 1 + log(count(t, d))    # if sublinear_tf=True
```
- Inverse Document Frequency (IDF):
```
idf = log((1 + N) / (1 + df(t))) + 1
```
where:

- N = total number of documents
- df(t) = number of documents containing term t
- Weight:
```
weight(t, d) = tf * idf
```
If `l2_norm=True`:
```
weight(t, d) = weight(t, d) / sqrt(sum(weight(t', d)^2))
```


### Example
```python
from vectorizer.tfidf import TfidfVectorizerScratch

docs = [["i", "love", "this", "movie"],
        ["i", "hate", "this", "movie"]]

vec = TfidfVectorizerScratch(sublinear_tf=False, l2_norm=False)
X = vec.fit_transform(docs)

print(vec.get_feature_names_out())  # ['hate', 'i', 'love', 'movie', 'this']
print(X.toarray().round(3))
```

Step-by-step for "love" in Doc 1
```
TF = 1
IDF = log((1 + 2) / (1 + 1)) + 1
    = log(3/2) + 1
    ≈ 0.405 + 1
    ≈ 1.405
Weight = 1 × 1.405 = 1.405
```
Vector
```
Doc 1: [0.000, 1.000, 1.405, 1.000, 1.000]
Doc 2: [1.405, 1.000, 0.000, 1.000, 1.000]
```

### Choosing Between BoW and TF-IDF

- BoW:
  - Pros: Simple, fast, works well for high-frequency tokens. 
  - Cons: Does not account for term importance across the corpus.

- TF-IDF:
  - Pros: Down-weights very common terms, highlights informative words.
  - Cons: Can be less interpretable; requires additional computation.

## References
- Why l2 norm in TF IDF?
  - https://medium.com/analytics-vidhya/understand-tf-idf-by-building-it-from-scratch-adc11eba7142
- Bag of Words
  - https://www.askpython.com/python/examples/bag-of-words-model-from-scratch