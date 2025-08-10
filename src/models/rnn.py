# rnn.py
import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    Minimal text classifier:
      Embedding -> RNN -> Dropout -> Linear(1)  (single logit)
    Train with BCEWithLogitsLoss. Apply sigmoid only at inference.
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bidirectional = bool(bidirectional)

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh",
        )

        out_dim = hidden_size * (2 if self.bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) int64 token ids (PAD=0)
        returns: (B,) raw logits
        """
        e = self.emb(x)                 # (B, T, E)
        # RNN outputs:
        #   out: (B, T, H * num_directions)
        #   h:   (L * num_directions, B, H)
        out, h = self.rnn(e)

        if self.bidirectional:
            # concat the last forward and last backward hidden states from the top layer
            # h[-2] = last forward, h[-1] = last backward
            last = torch.cat([h[-2], h[-1]], dim=1)   # (B, 2H)
        else:
            last = h[-1]                               # (B, H)

        z = self.drop(last)
        logit = self.fc(z).squeeze(1)                  # (B,)
        return logit
