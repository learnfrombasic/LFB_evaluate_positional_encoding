import torch

from src.models.layers.positional_encoding import KerplePositionalEncoding


def test_bias_shape():
    kerple = KerplePositionalEncoding(num_heads=4)
    bias = kerple.attention_bias(seq_len=10, device=torch.device("cpu"), dtype=torch.float32)
    assert bias.shape == (1, 4, 10, 10)


def test_bias_is_symmetric_in_distance():
    kerple = KerplePositionalEncoding(num_heads=3)
    bias = kerple.attention_bias(seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.allclose(bias, bias.transpose(-1, -2), atol=1e-6)


def test_bias_monotonic_decay_with_distance():
    kerple = KerplePositionalEncoding(num_heads=2)
    bias = kerple.attention_bias(seq_len=10, device=torch.device("cpu"), dtype=torch.float32)
    row = bias[0, 0, 0]  # bias from query position 0 to all key positions
    assert torch.all(row[1:] <= row[:-1] + 1e-6)


def test_r1_r2_are_learnable_and_receive_gradients():
    """Unlike ALiBi's fixed slopes, KERPLE's (r1, r2) must be trainable -
    the whole point of the scheme is that the decay shape is learned, not
    hand-picked."""
    kerple = KerplePositionalEncoding(num_heads=2)
    assert kerple.raw_r1.requires_grad
    assert kerple.raw_r2.requires_grad

    bias = kerple.attention_bias(seq_len=6, device=torch.device("cpu"), dtype=torch.float32)
    bias.sum().backward()
    assert kerple.raw_r1.grad is not None
    assert kerple.raw_r2.grad is not None
    assert torch.any(kerple.raw_r1.grad != 0)


def test_bias_changes_after_parameter_update():
    """Regression guard for the no-caching design: a stale cache (like
    ALiBi's, valid there because its slopes never change) would make this
    test fail."""
    kerple = KerplePositionalEncoding(num_heads=2)
    b1 = kerple.attention_bias(seq_len=6, device=torch.device("cpu"), dtype=torch.float32)
    with torch.no_grad():
        kerple.raw_r1.add_(1.0)
    b2 = kerple.attention_bias(seq_len=6, device=torch.device("cpu"), dtype=torch.float32)
    assert not torch.allclose(b1, b2)
