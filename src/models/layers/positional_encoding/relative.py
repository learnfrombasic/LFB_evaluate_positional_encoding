"""Relative position representations (Shaw et al., 2018).

Reference: https://arxiv.org/abs/1803.02155

This is an attention-level scheme: a learnable, clipped relative-position
embedding table is added directly into the attention-score computation via
`attention_bias`, contributing the `x_i W^Q . a^K_{ij}` term from the paper.
The symmetric `a^V` term (a second table added to the value aggregation) is
omitted, matching the widely used simplified variant (e.g. HuggingFace BERT's
`relative_key` position embedding type) for efficiency - the K-side term is
the dominant contribution.

A prior version of this module instead averaged the whole pairwise relative
table down to one per-position vector and added it once at the embedding
layer, which throws away the pairwise (query, key) structure that makes this
scheme "relative" in the first place.
"""

import torch
import torch.nn as nn


class RelativePositionalEncoding(nn.Module):
    """Shaw et al. (2018) relative position bias for attention scores.

    Args:
        d_model: dimensionality of a single attention head (head_size) -
            the relative embedding table lives in per-head space.
        max_len: maximum sequence length (only used as an upper bound check).
        clipping_distance: relative offsets are clipped to
            [-clipping_distance, +clipping_distance], as in the paper.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        max_len: int = 5000,
        clipping_distance: int = 32,
    ):
        super().__init__()
        self.clipping_distance = clipping_distance
        self.max_len = max_len
        num_positions = 2 * clipping_distance + 1
        self.rel_k = nn.Parameter(torch.randn(num_positions, d_model) * 0.02)

    def _rel_indices(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device)
        rel = pos[None, :] - pos[:, None]  # (T, T): offset from query i to key j
        rel = rel.clamp(-self.clipping_distance, self.clipping_distance)
        return rel + self.clipping_distance  # shift to [0, 2*clipping_distance]

    def attention_bias(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns a (B, num_heads, T, T) bias to add to attention scores.

        Args:
            q: query tensor of shape (B, num_heads, T, head_size), required.
        """
        if q is None:
            raise ValueError("RelativePositionalEncoding requires `q` to compute its bias")

        rel_idx = self._rel_indices(seq_len, device)  # (T, T)
        a_k = self.rel_k.to(dtype)[rel_idx]  # (T, T, head_size)
        # e_ij += q_i . a_k[i, j]  (Shaw et al., eq. 4, K-side term)
        return torch.einsum("bhid,ijd->bhij", q, a_k)
