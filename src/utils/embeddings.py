from pathlib import Path
import torch
import numpy as np

def load_glove_embeddings(glove_filename, vocab, embedding_dim=100):
    # Get project root (src/utils → src → project root)
    project_root = Path(__file__).resolve().parents[2]
    glove_path = project_root / "data" / glove_filename

    if not glove_path.exists():
        raise FileNotFoundError(f"GloVe file not found at {glove_path}")

    # Read GloVe file
    embeddings_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    # Create embedding matrix
    embedding_matrix = np.random.normal(
        scale=0.6, size=(len(vocab), embedding_dim)
    ).astype(np.float32)

    for word, idx in vocab.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[idx] = vec

    return torch.tensor(embedding_matrix, dtype=torch.float32)
