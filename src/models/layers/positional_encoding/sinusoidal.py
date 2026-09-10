"""Fixed sinusoidal positional encoding (Vaswani et al., 2017).

Reference: https://arxiv.org/abs/1706.03762
tAPE mode reference (Foumani et al., 2023): https://arxiv.org/abs/2301.03526
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding, added at the embedding layer.

    Consolidates four historically duplicate classes:
        - FixedPositionalEncoding   -> batch_first=False
        - AbsolutePositionalEncoding -> batch_first=True, scale_factor
        - TemporalPositionalEncoding -> return_encoding_only=True
        - tAPE                       -> tAPE_mode=True

    Args:
        d_model: embedding dimension.
        dropout: dropout probability applied after adding encoding.
        max_len: maximum sequence length.
        scale_factor: scalar multiplier on the entire encoding table.
        tAPE_mode: if True, scales sin/cos argument by (d_model / max_len)
                   as in tAPE (Foumani et al., 2023).
        batch_first: if True expects (B, T, D); if False expects (T, B, D).
        return_encoding_only: if True returns the encoding tensor without
                              adding to x (equivalent to TemporalPositionalEncoding).
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        scale_factor: float = 1.0,
        tAPE_mode: bool = False,
        batch_first: bool = True,
        return_encoding_only: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.return_encoding_only = return_encoding_only

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        arg = position * div_term
        if tAPE_mode:
            arg = arg * (d_model / max_len)

        pe[:, 0::2] = torch.sin(arg)
        pe[:, 1::2] = torch.cos(arg)
        pe = scale_factor * pe.unsqueeze(0)  # (1, max_len, d_model) - batch-first
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            enc = self.pe[:, : x.size(1)]  # (1, T, D)
        else:
            enc = self.pe[:, : x.size(0)].permute(1, 0, 2)  # (T, 1, D)

        if self.return_encoding_only:
            return enc.expand(x.size(0), -1, -1)

        return self.dropout(x + enc)
