import nltk
from collections import Counter
import pandas as pd
from tqdm.notebook import tqdm  # Jupyter-friendly progress bar

def compute_pos_stats(texts: list, labels: list) -> pd.DataFrame:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    pos_counts = {0: Counter(), 1: Counter()}

    for text, s in tqdm(zip(texts, labels), total=len(texts), desc="POS tagging"):
        s = int(s)
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        pos_counts[s].update(tag for _, tag in tagged)

    rows = []
    for s in (1, 0):
        total = sum(pos_counts[s].values()) or 1
        for tag, cnt in pos_counts[s].most_common():
            rows.append({"sentiment": s, "pos": tag, "count": cnt, "pct": cnt / total})

    return pd.DataFrame(rows)