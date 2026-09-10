import torch

from src.models.layers.positional_encoding import T5RelativePositionalEncoding


def test_bias_shape_and_gradient_flows():
    t5 = T5RelativePositionalEncoding(num_heads=4, num_buckets=16, max_distance=32)
    bias = t5.attention_bias(seq_len=12, device=torch.device("cpu"), dtype=torch.float32)
    assert bias.shape == (1, 4, 12, 12)
    bias.sum().backward()
    assert t5.relative_attention_bias.weight.grad is not None


def test_bidirectional_distinguishes_left_right_context():
    t5 = T5RelativePositionalEncoding(num_heads=1, num_buckets=8, bidirectional=True)
    buckets = t5._buckets(seq_len=5, device=torch.device("cpu"))
    # position 0 attending to 4 (right) should differ from 4 attending to 0 (left)
    assert buckets[0, 4].item() != buckets[4, 0].item()
