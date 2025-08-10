import torch
import torch.nn as nn

class DNNBinary(nn.Module):

    def __init__(self, vocab_size, emb_dim=128, hidden=128, p=0.2, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.drop = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden, 1)  # single logit for binary

        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, T) token ids (PAD=0)
        e = self.emb(x)                              # (B, T, E)
        mask = (x != 0).unsqueeze(-1)               # (B, T, 1)
        e = e * mask                                 # zero-out PAD embeddings
        sum_e = e.sum(dim=1)                         # (B, E)
        lengths = mask.sum(dim=1).clamp_min(1)       # (B, 1)
        avg = sum_e / lengths                        # (B, E)

        h = self.act(self.fc1(avg))
        h = self.drop(h)
        logit = self.fc2(h).squeeze(1)               # (B,)
        return logit
