import torch

from src.models.layers.positional_encoding import LearnablePositionalEncoding


def test_output_shape_and_gradient_flows():
    pe = LearnablePositionalEncoding(d_model=8, max_len=16, dropout=0.0)
    x = torch.zeros(2, 5, 8, requires_grad=True)
    out = pe(x)
    assert out.shape == (2, 5, 8)
    out.sum().backward()
    assert pe.pe.grad is not None


def test_init_std_changes_scale():
    torch.manual_seed(0)
    pe_uniform = LearnablePositionalEncoding(d_model=32, max_len=16, init_std=0.0)
    pe_normal = LearnablePositionalEncoding(d_model=32, max_len=16, init_std=1.0)
    assert pe_uniform.pe.std().item() < pe_normal.pe.std().item()
