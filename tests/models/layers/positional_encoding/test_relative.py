import pytest
import torch

from src.models.layers.positional_encoding import RelativePositionalEncoding


def test_bias_shape():
    rel = RelativePositionalEncoding(d_model=8, max_len=32, clipping_distance=4)
    q = torch.randn(2, 3, 10, 8)
    bias = rel.attention_bias(seq_len=10, device=q.device, dtype=q.dtype, q=q)
    assert bias.shape == (2, 3, 10, 10)


def test_missing_q_raises():
    rel = RelativePositionalEncoding(d_model=8, max_len=32, clipping_distance=4)
    with pytest.raises(ValueError):
        rel.attention_bias(seq_len=10, device=torch.device("cpu"), dtype=torch.float32)


def test_relative_indices_are_clipped_and_symmetric():
    rel = RelativePositionalEncoding(d_model=8, max_len=32, clipping_distance=4)
    idx = rel._rel_indices(seq_len=20, device=torch.device("cpu"))
    assert idx.min().item() >= 0
    assert idx.max().item() <= 2 * rel.clipping_distance
    # diagonal (offset 0) must map to the center bucket
    assert torch.all(idx.diagonal() == rel.clipping_distance)
