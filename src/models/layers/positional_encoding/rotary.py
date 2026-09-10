"""Rotary Position Embedding (RoPE) (Su et al., 2021 - RoFormer).

Reference: https://arxiv.org/abs/2104.09864

RoPE is an attention-level scheme: it rotates the query/key vectors of each
attention head *inside* every layer, not the token embeddings once up front.
Applying it once at the embedding layer (as a prior version of this module
did) discards the rotation's relative-position property as soon as it passes
through the first linear projection, and does not re-inject any position
signal into deeper layers. `rotate_qk` is applied per attention layer, per
head, on head-sized query/key vectors, matching the RoFormer / GPT-NeoX
convention (rotate-half formulation).
"""

import torch
import torch.nn as nn


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding, applied to per-head query/key vectors.

    Args:
        d_model: dimensionality of a single attention head (head_size), not
            the full hidden size - RoPE is applied per head.
        max_len: maximum sequence length to precompute frequencies for
            (frequencies are still computed lazily per forward call to
            support any sequence length, this only documents intent).
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got {d_model}")
        self.d_model = d_model
        self.max_len = max_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, d_model/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, d_model)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def rotate_qk(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotates query/key tensors of shape (B, num_heads, T, head_size)."""
        seq_len = q.size(-2)
        cos, sin = self._cos_sin(seq_len, q.device, q.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_size)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot
