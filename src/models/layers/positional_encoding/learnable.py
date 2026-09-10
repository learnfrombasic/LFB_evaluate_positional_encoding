"""Learned positional encoding (Gehring et al., 2017).

Reference: https://arxiv.org/abs/1705.03122
"""

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """Learned positional encoding, added at the embedding layer.

    Consolidates two historically duplicate classes:
        - LearnedPositionalEncoding   -> init_std > 0  (randn-based init)
        - LearnablePositionalEncoding -> init_std == 0 (uniform(-0.02, 0.02) init)

    Args:
        d_model: embedding dimension.
        dropout: dropout probability.
        max_len: maximum sequence length.
        init_std: if > 0, initialises with N(0, init_std^2);
                  if == 0, uses Uniform(-0.02, 0.02).
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, d_model))
        if init_std > 0:
            nn.init.normal_(self.pe, std=init_std)
        else:
            nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects batch-first: (B, T, D)
        return self.dropout(x + self.pe[: x.size(1)])
