"""T5-style relative position bias (Raffel et al., 2019).

Reference: https://arxiv.org/abs/1910.10683
Adapted from the HuggingFace T5 implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

Bonus positional-encoding option (config key `t5_relative`): buckets relative
offsets on a log scale and looks up a learned, per-head scalar bias for each
bucket, added directly to the attention scores. This is an alternative,
bucketed relative scheme to Shaw et al.'s `relative` and was previously
implemented in this codebase but never wired into `get_pos_encoder`.
"""

import math

import torch
import torch.nn as nn


def _get_relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
) -> torch.Tensor:
    """Map relative positions to bucket indices using log-scale binning.

    Half the buckets cover exact small offsets; the other half cover
    logarithmically larger ranges up to max_distance.
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
    """Returns a (seq_len, seq_len) bucket-index tensor for T5-style bias."""
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return _get_relative_position_bucket(
        x - y, bidirectional, num_buckets, max_distance
    )


class T5RelativePositionalEncoding(nn.Module):
    """Attention-level T5-style bucketed relative position bias.

    Args:
        num_heads: number of attention heads (bias is per-head).
        num_buckets: number of log-scale relative-position buckets.
        max_distance: offsets beyond this are clamped into the last bucket.
        bidirectional: whether to distinguish left/right context (True for
            an encoder such as BERT; False for a causal decoder).
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        self._bucket_cache: dict[int, torch.Tensor] = {}

    def _buckets(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len not in self._bucket_cache:
            self._bucket_cache[seq_len] = get_relative_positions(
                seq_len, self.bidirectional, self.num_buckets, self.max_distance
            ).to(device)
        return self._bucket_cache[seq_len]

    def attention_bias(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns a (1, num_heads, T, T) bias to add to attention scores."""
        buckets = self._buckets(seq_len, device)  # (T, T)
        values = self.relative_attention_bias(buckets)  # (T, T, num_heads)
        return values.permute(2, 0, 1).unsqueeze(0).to(dtype)
