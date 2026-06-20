# The implementations refer to this: https://github.com/imics-lab/positional-encoding-benchmark/blob/main/src/encodings/positional_encodings.py
from typing import Type

import math

import torch
import torch.nn as nn


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class tAPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 1024,
        scale_factor: float = 1.0,
    ):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin((position * div_term) * (d_model / max_len))
        pe[:, 1::2] = torch.cos((position * div_term) * (d_model / max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) - used in models like LLaMA"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(RotaryPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Create frequency matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        device = x.device

        # Generate position indices
        position = torch.arange(seq_len, device=device).float()

        # Create frequency matrix for all positions
        freqs = torch.outer(position, self.inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)

        # Apply rotary embedding
        cos_freqs = freqs.cos()
        sin_freqs = freqs.sin()

        # Reshape for broadcasting
        cos_freqs = cos_freqs.unsqueeze(0).expand(x.shape[0], -1, -1)
        sin_freqs = sin_freqs.unsqueeze(0).expand(x.shape[0], -1, -1)

        # Apply rotation
        x_rotated = self.apply_rotary_pos_emb(x, cos_freqs, sin_freqs)
        return self.dropout(x_rotated)

    def apply_rotary_pos_emb(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # Split the last dimension in half
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Apply rotation
        rotated = torch.zeros_like(x)
        rotated[..., ::2] = x1 * cos[..., ::2] - x2 * sin[..., ::2]
        rotated[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]

        return rotated


class RelativePositionalEncoding(nn.Module):
    """Relative Positional Encoding - focuses on relative distances between tokens"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        # Learnable relative position embeddings
        self.relative_positions = nn.Parameter(
            torch.randn(2 * max_len - 1, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # Create relative position matrix
        positions = torch.arange(seq_len, device=x.device)
        relative_positions = positions[:, None] - positions[None, :]
        relative_positions += self.max_len - 1  # Shift to positive indices

        # Get relative position embeddings
        rel_pos_emb = self.relative_positions[relative_positions]

        # Average the relative position embeddings for each position
        pos_encoding = rel_pos_emb.mean(dim=1).unsqueeze(0).expand(batch_size, -1, -1)

        x = x + pos_encoding
        return self.dropout(x)


class AbsolutePositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 1024,
        scale_factor: float = 1.0,
    ):
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x)


def _get_relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
) -> torch.Tensor:
    """
    from https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(
            relative_position, torch.zeros_like(relative_position)
        )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_postion_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_postion_if_large = torch.min(
        relative_postion_if_large,
        torch.full_like(relative_postion_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(
        is_small, relative_position, relative_postion_if_large
    )
    return relative_buckets


def get_relative_positions(
    seq_len: int,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    relative_positions = _get_relative_position_bucket(
        x - y, bidirectional, num_buckets, max_distance
    )
    return relative_positions


class SineSPE(nn.Module):
    def __init__(self, in_features: int, max_len: int = 512):
        super(SineSPE, self).__init__()
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
        return self.sine[:seq_len, :].unsqueeze(0)  # Shape: (1, seq_len, in_features)


class ConvSPE(nn.Module):
    def __init__(
        self,
        num_heads: int,
        in_features: int,
        kernel_size: int = 3,
        num_realizations: int = 1,
    ):
        super(ConvSPE, self).__init__()
        padding = (
            kernel_size // 2
        )  # This ensures that the output size matches the input size if stride=1

        # Define a 1D convolutional layer
        self.conv = nn.Conv1d(
            in_features, in_features, kernel_size=kernel_size, padding=padding
        )

        self.num_heads = num_heads
        self.in_features = in_features
        self.kernel_size = kernel_size
        self.num_realizations = num_realizations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should be of shape (batch_size, seq_len, in_features)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, in_features, seq_len)
        x = self.conv(x)  # Apply convolution
        x = x.permute(
            0, 2, 1
        )  # Change shape back to (batch_size, seq_len, in_features)
        return x


# Temporal Positional Encoding (T-PE)
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 896):  # Assuming 896 timesteps
        super(TemporalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)


# Variable Positional Encoding for handling multivariate data
class VariablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_variables: int):
        super(VariablePositionalEncoding, self).__init__()
        self.variable_embedding = nn.Embedding(num_variables, d_model)

    def forward(self, x: torch.Tensor, variable_idx: torch.Tensor) -> torch.Tensor:
        variable_embed = self.variable_embedding(variable_idx)
        return x + variable_embed.unsqueeze(0)


def get_pos_encoder(pos_encoding: str) -> Type[nn.Module]:
    if pos_encoding == "fixed":
        return FixedPositionalEncoding
    elif pos_encoding == "learned":
        return LearnedPositionalEncoding
    elif pos_encoding == "tape":
        return tAPE
    elif pos_encoding == "absolute":
        return AbsolutePositionalEncoding
    else:
        raise ValueError(f"Unknown positional encoding type: {pos_encoding}")
