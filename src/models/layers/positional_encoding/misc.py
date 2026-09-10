"""Extra positional-encoding building blocks not wired into BERT's
`get_pos_encoder` registry.

These target multivariate/time-series use cases (SPE, per-channel encoding)
rather than text, and are kept here for potential future reuse. Moved
verbatim from the original monolithic `positional_encoding.py`.

Reference (SPE): Liutkus et al. (2021) - https://arxiv.org/abs/2108.12409
"""

import math

import torch
import torch.nn as nn


class SineSPE(nn.Module):
    """Sinusoidal Stochastic Positional Encoding (sine variant)."""

    def __init__(self, in_features: int, max_len: int = 512):
        super().__init__()
        self.in_features = in_features
        self.max_len = max_len
        self.position = nn.Parameter(torch.zeros(1, max_len, in_features))
        self.register_buffer("sine", self._generate_sine_encoding())

    def _generate_sine_encoding(self) -> torch.Tensor:
        position = torch.arange(self.max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.in_features, 2).float()
            * -(math.log(10000.0) / self.in_features)
        )
        encoding = torch.zeros(self.max_len, self.in_features)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns (1, seq_len, in_features) encoding tensor."""
        return self.sine[:seq_len, :].unsqueeze(0)


class ConvSPE(nn.Module):
    """Convolutional Stochastic Positional Encoding."""

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        kernel_size: int = 3,
        num_realizations: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_features, in_features, kernel_size=kernel_size, padding=padding
        )
        self.num_heads = num_heads
        self.in_features = in_features
        self.kernel_size = kernel_size
        self.num_realizations = num_realizations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> conv expects (B, D, T)
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)


class VariablePositionalEncoding(nn.Module):
    """Per-variable (channel) positional encoding for multivariate sequences.

    Adds a learned embedding for each input variable/channel, allowing the
    model to distinguish between different sensor streams or feature
    channels. No dedicated paper; standard embedding-table approach.
    """

    def __init__(self, d_model: int, num_variables: int):
        super().__init__()
        self.variable_embedding = nn.Embedding(num_variables, d_model)

    def forward(self, x: torch.Tensor, variable_idx: torch.Tensor) -> torch.Tensor:
        return x + self.variable_embedding(variable_idx).unsqueeze(0)
