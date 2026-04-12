"""Shared LSTM direction classifier (optional interval_id embedding)."""
from __future__ import annotations

import torch
from torch import nn


class LSTMDir(nn.Module):
    """
    If num_intervals is set, `interval_id` is embedded and concatenated to the LSTM
    pooled vector; the LSTM input should NOT include the raw interval_id column.
    Legacy checkpoints use num_intervals=None and pass all features (including float interval_id) in x.
    """

    def __init__(
        self,
        d_in: int,
        hid: int = 128,
        att: bool = False,
        *,
        num_intervals: int | None = None,
        embed_dim: int = 8,
    ):
        super().__init__()
        self.att = att
        self.num_intervals = num_intervals
        self.lstm = nn.LSTM(d_in, hid, batch_first=True)
        if att:
            self.w = nn.Linear(hid, 1, bias=False)
        self.interval_emb: nn.Embedding | None
        if num_intervals is not None:
            self.interval_emb = nn.Embedding(num_intervals, embed_dim)
            self.fc = nn.Linear(hid + embed_dim, 1)
        else:
            self.interval_emb = None
            self.fc = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor, interval_id: torch.Tensor | None = None) -> torch.Tensor:
        o, _ = self.lstm(x)
        if self.att:
            a = torch.softmax(self.w(o), dim=1)
            o = (a * o).sum(1)
        else:
            o = o[:, -1]
        if self.interval_emb is not None:
            if interval_id is None:
                raise ValueError("interval_id is required when num_intervals is set")
            e = self.interval_emb(interval_id.long())
            o = torch.cat([o, e], dim=-1)
        return self.fc(o)


def lstm_feature_columns(
    trained_feats: list[str], use_interval_embedding: bool
) -> tuple[list[str], bool]:
    """Return (columns_for_lstm_input, use_embedding)."""
    if use_interval_embedding and "interval_id" in trained_feats:
        return [c for c in trained_feats if c != "interval_id"], True
    return list(trained_feats), False
