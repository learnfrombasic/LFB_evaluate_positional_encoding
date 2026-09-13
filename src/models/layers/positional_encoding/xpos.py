"""xPos - Extrapolatable Position Embedding (Sun et al., 2022 - "A Length-
Extrapolatable Transformer").

Reference: https://arxiv.org/abs/2212.10554

xPos extends RoPE with a per-dimension exponential decay: alongside the
standard rotation, the query vector at position n is scaled by `zeta^n` and
the key vector at position m by `zeta^-m`, for a per-dimension-pair base
`zeta < 1`. In the q.k dot product the two scale factors combine into
`zeta^(n-m)`: a smooth, learned-free decay purely as a function of relative
distance, layered on top of RoPE's rotation. The paper reports this improves
extrapolation to sequences longer than training versus plain RoPE.

Positions are centered (`position - seq_len // 2`) before computing the
scale exponent purely to keep `zeta^exponent` numerically bounded for long
sequences; centering cancels out of every `n - m` difference, so it does not
change which relative distance the decay actually depends on (verified by
`tests/.../test_xpos.py::test_relative_position_invariance_of_rotation`).

Caveat: the paper targets causal (decoder-only) models, where `n >= m`
always and the decay is one-directional. Applied to this project's
bidirectional attention, the same `zeta^(n-m)` factor decays scores for
`n > m` but *grows* them for `n < m` - an asymmetric effect the paper does
not study directly. It is included here as-is, as one more attention-level
scheme to compare, not as a claim that it behaves optimally in a
bidirectional encoder.
"""

import torch
import torch.nn as nn


class XPosPositionalEncoding(nn.Module):
    """xPos: RoPE rotation plus a relative-distance decay on Q/K magnitude.

    Args:
        d_model: dimensionality of a single attention head (head_size).
        max_len: unused directly (frequencies/scales are computed lazily per
            forward call); kept for interface parity with RoPE/Relative.
        scale_base: length scale over which the decay exponent grows by one
            unit; larger values -> slower decay. 512 matches the paper's
            reference implementation.
        gamma: shifts the per-dimension decay base away from 0, so no
            dimension has an exactly-1.0 (no decay) or exactly-0.0
            (instant collapse) base. 0.4 matches the paper.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        max_len: int = 5000,
        scale_base: float = 512.0,
        gamma: float = 0.4,
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"xPos requires an even head dimension, got {d_model}")
        self.d_model = d_model
        self.max_len = max_len
        self.scale_base = scale_base

        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Per-dimension-pair decay base in (gamma/(1+gamma), 1), one value per
        # frequency; duplicated to match the [freqs, freqs] layout used by
        # rotate-half, exactly like RoPE's cos/sin construction.
        idx = torch.arange(0, d_model, 2).float()
        base = (idx + gamma * d_model) / ((1 + gamma) * d_model)
        self.register_buffer("scale_base_per_dim", torch.cat([base, base]), persistent=False)

    def _cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, d_model/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, d_model)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _qk_scale(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Returns (T, d_model) of zeta^(centered position / scale_base)."""
        pos = torch.arange(seq_len, device=device, dtype=torch.float32) - seq_len // 2
        exponent = (pos / self.scale_base).unsqueeze(-1)  # (T, 1)
        base = self.scale_base_per_dim.to(device=device, dtype=torch.float32)  # (d_model,)
        return (base.unsqueeze(0) ** exponent).to(dtype)  # (T, d_model)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def rotate_qk(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotates + scales query/key tensors of shape (B, num_heads, T, head_size)."""
        seq_len = q.size(-2)
        cos, sin = self._cos_sin(seq_len, q.device, q.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_size)
        sin = sin.unsqueeze(0).unsqueeze(0)

        scale = self._qk_scale(seq_len, q.device, q.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_size)
        q_scaled = q * scale
        k_scaled = k / scale

        q_rot = q_scaled * cos + self._rotate_half(q_scaled) * sin
        k_rot = k_scaled * cos + self._rotate_half(k_scaled) * sin
        return q_rot, k_rot
