"""
Positional Encoding implementations for Transformer models.

References:
    - Original Transformer (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762
    - Learnable PE (Gehring et al., 2017): https://arxiv.org/abs/1705.03122
    - tAPE (Foumani et al., 2023): https://arxiv.org/abs/2301.03526
    - RoPE (Su et al., 2021): https://arxiv.org/abs/2104.09864
    - Relative PE (Shaw et al., 2018): https://arxiv.org/abs/1803.02155
    - T5 relative buckets (Raffel et al., 2019): https://arxiv.org/abs/1910.10683
    - SPE (Liutkus et al., 2021): https://arxiv.org/abs/2108.12409
    - Benchmark repo: https://github.com/imics-lab/positional-encoding-benchmark
"""

import math
from functools import partial

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Sinusoidal PE (consolidates: FixedPositionalEncoding, AbsolutePositionalEncoding,
#                               TemporalPositionalEncoding, tAPE)
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Consolidates four previously duplicate classes:
        - FixedPositionalEncoding   → batch_first=False
        - AbsolutePositionalEncoding → batch_first=True, scale_factor
        - TemporalPositionalEncoding → return_encoding_only=True
        - tAPE                       → tAPE_mode=True

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
        pe = scale_factor * pe.unsqueeze(0)  # (1, max_len, d_model) — batch-first
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            enc = self.pe[:, : x.size(1)]  # (1, T, D)
        else:
            enc = self.pe[:, : x.size(0)].permute(1, 0, 2)  # (T, 1, D)

        if self.return_encoding_only:
            return enc.expand(x.size(0), -1, -1)

        return self.dropout(x + enc)


# ---------------------------------------------------------------------------
# Learnable PE (consolidates: LearnedPositionalEncoding, LearnablePositionalEncoding)
# ---------------------------------------------------------------------------


class LearnablePositionalEncoding(nn.Module):
    """Learned positional encoding (Gehring et al., 2017).

    Consolidates two previously duplicate classes:
        - LearnedPositionalEncoding   → init_std > 0  (randn-based init)
        - LearnablePositionalEncoding → init_std == 0 (uniform(-0.02, 0.02) init)

    Args:
        d_model: embedding dimension.
        dropout: dropout probability.
        max_len: maximum sequence length.
        init_std: if > 0, initialises with N(0, init_std²);
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


# ---------------------------------------------------------------------------
# Rotary Positional Encoding — RoPE (Su et al., 2021)
# https://arxiv.org/abs/2104.09864
# Used in: LLaMA, PaLM, GPT-NeoX
# ---------------------------------------------------------------------------


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Unlike additive encodings, RoPE rotates query/key vectors in attention,
    encoding relative positions via rotation matrices. This gives exact
    relative position information without a separate relative-bias table.

    Reference: Su et al. (2021) — https://arxiv.org/abs/2104.09864
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        device = x.device

        position = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(position, self.inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)

        cos_freqs = freqs.cos().unsqueeze(0).expand(x.shape[0], -1, -1)
        sin_freqs = freqs.sin().unsqueeze(0).expand(x.shape[0], -1, -1)

        return self.dropout(self._apply_rotary(x, cos_freqs, sin_freqs))

    def _apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.zeros_like(x)
        rotated[..., ::2] = x1 * cos[..., ::2] - x2 * sin[..., ::2]
        rotated[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        return rotated


# ---------------------------------------------------------------------------
# Relative Positional Encoding (Shaw et al., 2018)
# https://arxiv.org/abs/1803.02155
# ---------------------------------------------------------------------------


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding via learnable pairwise distance embeddings.

    Encodes the relative offset (i - j) between every pair of positions.
    The per-position encoding is the mean of all relative embeddings pointing
    to that position, added to the input as a bias.

    Reference: Shaw et al. (2018) — https://arxiv.org/abs/1803.02155
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        # Table covers offsets in [-(max_len-1), +(max_len-1)]
        self.relative_positions = nn.Parameter(
            torch.randn(2 * max_len - 1, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        positions = torch.arange(seq_len, device=x.device)
        rel_idx = positions[:, None] - positions[None, :]  # (T, T)
        rel_idx = rel_idx + self.max_len - 1  # shift to [0, 2*max_len-2]

        rel_pos_emb = self.relative_positions[rel_idx]  # (T, T, D)
        pos_encoding = (
            rel_pos_emb.mean(dim=1)  # (T, D)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        return self.dropout(x + pos_encoding)


# ---------------------------------------------------------------------------
# T5-style relative position buckets (Raffel et al., 2019)
# https://arxiv.org/abs/1910.10683
# ---------------------------------------------------------------------------


def _get_relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
) -> torch.Tensor:
    """Map relative positions to bucket indices using log-scale binning.

    Half the buckets cover exact small offsets; the other half cover
    logarithmically larger ranges up to max_distance.

    Adapted from the HuggingFace T5 implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

    Reference: Raffel et al. (2019) — https://arxiv.org/abs/1910.10683
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

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets


def get_relative_positions(
    seq_len: int,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """Return a (seq_len, seq_len) bucket-index tensor for T5-style bias."""
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return _get_relative_position_bucket(
        x - y, bidirectional, num_buckets, max_distance
    )


# ---------------------------------------------------------------------------
# Stochastic Positional Encoding — SPE (Liutkus et al., 2021)
# https://arxiv.org/abs/2108.12409
# ---------------------------------------------------------------------------


class SineSPE(nn.Module):
    """Sinusoidal Stochastic Positional Encoding (sine variant).

    Reference: Liutkus et al. (2021) — https://arxiv.org/abs/2108.12409
    """

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
    """Convolutional Stochastic Positional Encoding.

    Reference: Liutkus et al. (2021) — https://arxiv.org/abs/2108.12409
    """

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
        # x: (B, T, D) → conv expects (B, D, T)
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Variable Positional Encoding — for multivariate / channel-wise data
# ---------------------------------------------------------------------------


class VariablePositionalEncoding(nn.Module):
    """Per-variable (channel) positional encoding for multivariate sequences.

    Adds a learned embedding for each input variable/channel, allowing the
    model to distinguish between different sensor streams or feature channels.
    No dedicated paper; standard embedding-table approach.
    """

    def __init__(self, d_model: int, num_variables: int):
        super().__init__()
        self.variable_embedding = nn.Embedding(num_variables, d_model)

    def forward(self, x: torch.Tensor, variable_idx: torch.Tensor) -> torch.Tensor:
        return x + self.variable_embedding(variable_idx).unsqueeze(0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_pos_encoder(pos_encoding: str):
    mapping = {
        "sinusoidal": SinusoidalPositionalEncoding,
        "fixed": SinusoidalPositionalEncoding,
        "absolute": SinusoidalPositionalEncoding,
        "temporal": partial(SinusoidalPositionalEncoding, return_encoding_only=True),
        "tape": partial(SinusoidalPositionalEncoding, tAPE_mode=True),
        "learnable": LearnablePositionalEncoding,
        "learned": partial(LearnablePositionalEncoding, init_std=1.0),
        "rotary": RotaryPositionalEncoding,
        "rope": RotaryPositionalEncoding,
        "relative": RelativePositionalEncoding,
    }
    key = pos_encoding.lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown positional encoding: '{pos_encoding}'. "
            f"Valid options: {sorted(mapping)}"
        )
    return mapping[key]
