import torch

from src.models.layers.positional_encoding.alibi import (
    ALiBiPositionalEncoding,
    _alibi_slopes,
)


def test_slopes_power_of_two_are_geometric():
    slopes = _alibi_slopes(8)
    assert len(slopes) == 8
    ratios = [slopes[i + 1] / slopes[i] for i in range(len(slopes) - 1)]
    assert all(abs(r - ratios[0]) < 1e-6 for r in ratios)


def test_slopes_count_matches_num_heads_when_not_power_of_two():
    for n in (3, 5, 6, 7, 12):
        assert len(_alibi_slopes(n)) == n


def test_bias_shape_and_monotonic_decay_with_distance():
    alibi = ALiBiPositionalEncoding(num_heads=4)
    bias = alibi.attention_bias(seq_len=10, device=torch.device("cpu"), dtype=torch.float32)
    assert bias.shape == (1, 4, 10, 10)
    row = bias[0, 0, 0]  # bias from query position 0 to all key positions
    # distance grows monotonically along the row -> bias must be non-increasing
    assert torch.all(row[1:] <= row[:-1] + 1e-6)


def test_bias_is_cached_per_seq_len():
    alibi = ALiBiPositionalEncoding(num_heads=2)
    b1 = alibi.attention_bias(seq_len=6, device=torch.device("cpu"), dtype=torch.float32)
    b2 = alibi.attention_bias(seq_len=6, device=torch.device("cpu"), dtype=torch.float32)
    assert b1 is b2
