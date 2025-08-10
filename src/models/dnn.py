import torch
import torch.nn as nn

class DNN(nn.Module):

    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # dropout layer
        self.fc2 = nn.Linear(hidden_dim, 1)  # binary classification

    def forward(self, x):
        emb = self.embedding(x)  # (B, L, E)
        avg_emb = emb.mean(dim=1)
        h = self.fc1(avg_emb)
        h = self.relu(h)
        h = self.dropout(h)  # apply dropout
        out = self.fc2(h).squeeze(1)  # (B,)

        return out
