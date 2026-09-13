"""KERPLE - Kernelized Relative Positional Embeddings for Length Extrapolation
(Chi et al., 2022).

Reference: https://arxiv.org/abs/2205.09921

KERPLE generalizes ALiBi: instead of a fixed, non-learnable geometric slope
per head applied to a linear distance penalty (`-m_h * |i-j|`), it learns a
per-head scale `r1_h` and a per-head rate `r2_h` and applies them through a
logarithmic kernel: `bias_h(i, j) = -r1_h * log(1 + r2_h * |i-j|)`. ALiBi's
fixed linear penalty is the special case of a kernel with a hand-picked
slope and no log-compression; here both parameters are learned from data,
and the log kernel grows sub-linearly, which is the paper's stated reason it
extrapolates to sequences longer than training even more robustly than
ALiBi's linear penalty.

Unlike ALiBi's bias (cached per sequence length, since it is a fixed
function of position with no learnable state), this bias must be
recomputed on every forward call: `r1`/`r2` are `nn.Parameter`s updated by
the optimizer every step, so a cached tensor would silently go stale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KerplePositionalEncoding(nn.Module):
    """Attention-level KERPLE bias (logarithmic variant).

    Args:
        num_heads: number of attention heads (one learnable (r1, r2) pair
            per head).
        eps: small floor added to the softplus-transformed r1/r2 so both
            stay strictly positive (a non-positive r1 would flip the bias
            into a reward for distance; a non-positive r2 would make the
            kernel non-monotonic).
    """

    def __init__(self, num_heads: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        # Raw, unconstrained parameters; softplus at use-time keeps r1/r2 > 0.
        # Zero init -> softplus(0) = ln(2) ~ 0.69, a mild starting slope/rate,
        # matching the paper's use of a modest initial bias strength.
        self.raw_r1 = nn.Parameter(torch.zeros(num_heads))
        self.raw_r2 = nn.Parameter(torch.zeros(num_heads))

    def attention_bias(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns a (1, num_heads, T, T) bias to add to attention scores."""
        positions = torch.arange(seq_len, device=device)
        distance = (positions[None, :] - positions[:, None]).abs().to(dtype)  # (T, T)

        r1 = (F.softplus(self.raw_r1) + self.eps).to(dtype).view(-1, 1, 1)
        r2 = (F.softplus(self.raw_r2) + self.eps).to(dtype).view(-1, 1, 1)

        bias = -r1 * torch.log1p(r2 * distance.unsqueeze(0))  # (num_heads, T, T)
        return bias.unsqueeze(0)  # (1, num_heads, T, T)
