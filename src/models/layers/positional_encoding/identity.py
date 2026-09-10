"""No positional encoding (NoPE) - the scientific control condition.

References:
    - Haviv et al. (2022), "Transformer Language Models without Positional
      Encodings Still Learn Positional Information" - https://arxiv.org/abs/2203.16634
    - Kazemnejad et al. (2023), "The Impact of Positional Encoding on Length
      Generalization in Transformers" - https://arxiv.org/abs/2305.19466

Adds no positional signal anywhere - not to the embedding, not to
attention. Every other scheme in this project should be read relative to
this baseline; without it, "PE choice matters" has no zero point to compare
against.

The literature's key distinction, and the reason this scheme is expected to
behave very differently here than in the causal (decoder-only) LMs most
NoPE papers study: a *causal* transformer's attention mask alone lets a
token infer how many tokens precede it, so causal NoPE models can be
competitive - even favored for length extrapolation, since there is no
positional table to run out of. This project's BERT is *bidirectional*
(full self-attention, no causal mask): with no positional signal anywhere,
the whole stack is permutation-equivariant - shuffle the input tokens and
every per-token output shuffles identically, so the model has no way to
distinguish word order at all. `tests/models/layers/positional_encoding/test_identity.py`
verifies this exact property empirically on the full model.
"""

import torch
import torch.nn as nn


class NoPositionalEncoding(nn.Module):
    """Identity: adds nothing. See module docstring for why this matters."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
