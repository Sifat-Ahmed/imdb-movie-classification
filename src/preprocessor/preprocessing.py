import re
import string
from typing import List
from .constants import CONTRACTIONS

import nltk
from nltk.corpus import stopwords

try:
    import emoji as _emoji
except Exception:
    _emoji = None

CONTRACTIONS_RE = re.compile(r"\b(" + "|".join(map(re.escape, CONTRACTIONS.keys())) + r")\b")


class IMDBPreprocessor:
    def __init__(self, language: str = "english"):
        self.stop = set(stopwords.words(language))
        self.punct_table = str.maketrans("", "", string.punctuation)

    def clean_text(self, text: str) -> str:
        s = re.sub(r'\\n', ' ', text)  # remove literal \n
        s = s.replace('\n', ' ')       # remove real newlines
        s = re.sub(r'\s+', ' ', s)     # collapse spaces
        return s.strip()

    def remove_html_tags(self, text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    def handle_emojis(self, text: str) -> str:
        if _emoji is None:
            return text
        return _emoji.demojize(text, language="en")

    def remove_punctuations(self, text: str) -> str:
        return text.translate(self.punct_table)

    def remove_stop_words(self, text: str) -> str:
        tokens = [t for t in text.split() if t.lower() not in self.stop]
        return " ".join(tokens)

    def normalize_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def expand_contractions(self, text: str) -> str:
        return CONTRACTIONS_RE.sub(lambda m: CONTRACTIONS[m.group(0)], text)

    def to_lower(self, text: str) -> str:
        return text.lower()

    def normalize_ratings(self, text: str) -> str:
        """Replace n/10 ratings with categorical flags."""
        def repl(m):
            n = int(m.group(1))
            if n >= 7:
                return " RATING_POS "
            elif n <= 3:
                return " RATING_NEG "
            else:  # 4,5,6
                return " RATING_NEUTRAL "
        return re.sub(r"\b([0-9]{1,2})\s*/\s*10\b", repl, text)

    def preprocess(self, text: str) -> str:
        x = self.remove_html_tags(text)
        x = self.clean_text(x)
        x = self.handle_emojis(x)
        x = self.normalize_ratings(x)   # <-- added here before punctuation removal
        x = self.to_lower(x)
        x = self.expand_contractions(x)
        x = self.remove_punctuations(x)
        x = self.remove_stop_words(x)
        x = self.normalize_spaces(x)
        return x

    def preprocess_many(self, texts: List[str]) -> List[str]:
        return [self.preprocess(t) for t in texts]
