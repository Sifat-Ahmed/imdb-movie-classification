import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=1,
                 bidirectional=False, dropout=0.4, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.bidirectional = bool(bidirectional)

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_dim * (2 if self.bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        # x: (B, L)
        emb = self.embedding(x)  # (B, L, E)
        out, (h, c) = self.lstm(emb)  # h: (num_layers*dir, B, H)

        if self.bidirectional:
            # concat last forward and last backward states from top layer
            last = torch.cat([h[-2], h[-1]], dim=1)  # (B, 2H)
        else:
            last = h[-1]  # (B, H)

        last = self.dropout(last)
        logit = self.fc(last).squeeze(1)  # (B,)
        return logit


class GloVeLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, bidirectional, dropout, pad_idx,
                 pretrained_embeddings=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # freeze GloVe weights

        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        return self.fc(self.dropout(hidden)).squeeze(1)