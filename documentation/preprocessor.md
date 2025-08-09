# Preprocessor Module Documentation

This document details the preprocessing logic implemented in the four scripts under 
```
├── src/
│   ├── preprocessor/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── pos_tag.py
│   │   └── preprocessing.py
```
These modules work together to clean and normalize IMDB review text before feature extraction and modeling.


### `constants.py`

Defines fixed constants used in preprocessing.

- A dictionary mapping English contractions to their expanded forms (e.g., "don't" → "do not").
- Ensures consistent expansion across all text inputs.

### `pos_tag.py`
Provides part-of-speech tagging utilities.

- Uses NLTK's POS tagger to assign grammatical categories to each token.
- Calculates POS tag distribution per sentiment label for exploratory analysis.
- Useful for linguistic insights and potential feature engineering.

### `preprocessing.py`
Contains the main preprocessing pipeline in `IMDBPreprocessor`.

#### Key Functions
- Cleans a single text string.
- Applies preprocess to a list of strings.

#### Cleaning Steps

- HTML Removal: Strips HTML tags.
- Newline Normalization: Removes literal and actual newlines.
- Emoji Handling: Converts emojis to text labels (if emoji package installed).
- Rating Normalization: Detects numeric ratings like 8/10 and replaces with tokens:
```
RATING_POS for ratings ≥ 7
RATING_NEG for ratings ≤ 3
RATING_NEUTRAL otherwise
```
- Lowercasing: Converts all text to lowercase.
- Contraction Expansion: Uses CONTRACTIONS mapping.
- Punctuation Removal: Strips non-alphanumeric characters.
- Stopword Removal: Removes English stopwords using NLTK.
- Whitespace Normalization: Collapses multiple spaces into one.

#### Why These Steps?

- Reduce noise and unify representation.
- Preserve sentiment-indicative patterns (e.g., ratings) while removing irrelevant symbols.
- Lowercase and contraction expansion improve vocabulary consistency.
- Stopword and punctuation removal reduce feature dimensionality.