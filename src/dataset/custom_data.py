import re
import string
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
from nltk.corpus import stopwords

try:
    import emoji as _emoji
except Exception:
    _emoji = None

PAD, UNK = "<PAD>", "<UNK>"

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, max_len=256, preprocess=True, min_freq=2, max_vocab_size=30000, language="english"):
        """
        :param texts: list/Series of strings
        :param labels: list/Series of ints
        :param max_len: max token length for each sample
        :param preprocess: if True, run built-in preprocessing
        :param min_freq: min token frequency to keep in vocab
        :param max_vocab_size: max vocab size (incl PAD, UNK)
        """
        self.texts = [str(t) for t in texts]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.max_len = max_len
        self.stop = set(stopwords.words(language))
        self.punct_table = str.maketrans("", "", string.punctuation)

        if preprocess:
            self.texts = [self._preprocess_text(t) for t in self.texts]

        # Build vocab after preprocessing
        self.vocab = self._build_vocab(self.texts, min_freq, max_vocab_size)

    # -----------------------------
    # Private preprocessing methods
    # -----------------------------
    def _remove_html_tags(self, text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    def _clean_newlines(self, text: str) -> str:
        s = re.sub(r'\\n', ' ', text)  # remove literal \n
        s = s.replace('\n', ' ')
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    def _handle_emojis(self, text: str) -> str:
        if _emoji is None:
            return text
        return _emoji.demojize(text, language="en")

    def _replace_ratings(self, text: str) -> str:
        # Replace numeric ratings like 8/10 with sentiment tokens
        def repl(m):
            try:
                score = int(m.group(1))
                if score >= 7:
                    return " RATING_POS "
                elif score <= 3:
                    return " RATING_NEG "
                else:
                    return " RATING_NEUTRAL "
            except:
                return m.group(0)
        return re.sub(r"\b(\d{1,2})/10\b", repl, text)

    def _to_lower(self, text: str) -> str:
        return text.lower()

    def _expand_contractions(self, text: str) -> str:
        from preprocessor.constants import CONTRACTIONS  # assuming same dict from earlier
        CONTRACTIONS_RE = re.compile(r"\b(" + "|".join(map(re.escape, CONTRACTIONS.keys())) + r")\b")
        return CONTRACTIONS_RE.sub(lambda m: CONTRACTIONS[m.group(0)], text)

    def _remove_punctuations(self, text: str) -> str:
        return text.translate(self.punct_table)

    def _remove_stop_words(self, text: str) -> str:
        tokens = [t for t in text.split() if t.lower() not in self.stop]
        return " ".join(tokens)

    def _normalize_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _preprocess_text(self, text: str) -> str:
        x = self._remove_html_tags(text)
        x = self._clean_newlines(x)
        x = self._handle_emojis(x)
        x = self._replace_ratings(x)
        x = self._to_lower(x)
        x = self._expand_contractions(x)
        x = self._remove_punctuations(x)
        #x = self._remove_stop_words(x)
        x = self._normalize_spaces(x)
        return x

    # -----------------------------
    # Vocab + encoding
    # -----------------------------
    def _build_vocab(self, texts, min_freq=2, max_size=30000):
        cnt = Counter()
        for t in texts:
            cnt.update(t.split())
        items = [(tok, c) for tok, c in cnt.items() if c >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        if max_size:
            items = items[: max(0, max_size - 2)]
        vocab = {PAD:0, UNK:1}
        for tok, _ in items:
            vocab[tok] = len(vocab)
        return vocab

    def _encode(self, text):
        toks = text.split()
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in toks[:self.max_len]]
        if len(ids) < self.max_len:
            ids += [self.vocab[PAD]] * (self.max_len - len(ids))
        return np.array(ids, dtype=np.int64)

    # -----------------------------
    # Dataset API
    # -----------------------------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self._encode(self.texts[idx])
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

if __name__ == "__main__":
    data = ["I love 33 this movie, so much_1", "I Didn't LIKE at ALL"]
    label = [1, 0]

    dataset = IMDBDataset(texts=data, labels=label, max_len=12, min_freq=1)

    for d in dataset:
        print(d)