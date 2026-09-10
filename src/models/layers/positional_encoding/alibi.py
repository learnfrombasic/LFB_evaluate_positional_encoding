"""ALiBi - Attention with Linear Biases (Press et al., 2021).

Reference: https://arxiv.org/abs/2108.12409

ALiBi does not modify token embeddings. Instead, it adds a static,
per-head linear bias directly to the attention scores before the softmax:
`scores_h += -m_h * |i - j|`, where `m_h` is a fixed, head-specific slope in
a geometric sequence. Slopes are computed once at construction time
(previously recomputed on every forward call) and the resulting bias is
cached per sequence length since padding keeps `seq_len` constant per run.
"""

import math

import torch
import torch.nn as nn


def _alibi_slopes(num_heads: int) -> list[float]:
    """Computes ALiBi slopes for an arbitrary number of heads.

    Matches the reference implementation from Press et al. (2021): for a
    power-of-two head count the slopes form a geometric sequence starting at
    2^(-8/num_heads); for any other head count, slopes for the next-lower
    power of two are used and the remainder are interpolated from the next
    power-of-two level.
    """

    def slopes_for_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return slopes_for_power_of_2(num_heads)

    closest_pow2 = 2 ** math.floor(math.log2(num_heads))
    extra = _alibi_slopes(2 * closest_pow2)[0::2][: num_heads - closest_pow2]
    return slopes_for_power_of_2(closest_pow2) + extra


class ALiBiPositionalEncoding(nn.Module):
    """Attention-level ALiBi bias.

    Args:
        num_heads: number of attention heads (slopes are one-per-head).
    """

    def __init__(self, num_heads: int):
        super().__init__()
        slopes = torch.tensor(_alibi_slopes(num_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)
        self._bias_cache: dict[tuple, torch.Tensor] = {}

    def attention_bias(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns a (1, num_heads, T, T) bias to add to attention scores."""
        cache_key = (seq_len, device, dtype)
        if cache_key not in self._bias_cache:
            positions = torch.arange(seq_len, device=device)
            distance = (positions[None, :] - positions[:, None]).abs().to(dtype)
            slopes = self.slopes.to(device=device, dtype=dtype).view(-1, 1, 1)
            self._bias_cache[cache_key] = (-distance.unsqueeze(0) * slopes).unsqueeze(0)
        return self._bias_cache[cache_key]
