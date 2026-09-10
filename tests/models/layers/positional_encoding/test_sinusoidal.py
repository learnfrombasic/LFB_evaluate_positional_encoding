import torch

from src.models.layers.positional_encoding import SinusoidalPositionalEncoding


def test_output_shape_and_finite():
    pe = SinusoidalPositionalEncoding(d_model=16, max_len=32, dropout=0.0)
    x = torch.zeros(2, 10, 16)
    out = pe(x)
    assert out.shape == (2, 10, 16)
    assert torch.isfinite(out).all()


def test_tape_mode_changes_the_encoding():
    pe_default = SinusoidalPositionalEncoding(d_model=16, max_len=32)
    pe_tape = SinusoidalPositionalEncoding(d_model=16, max_len=32, tAPE_mode=True)
    assert not torch.allclose(pe_default.pe, pe_tape.pe)


def test_return_encoding_only_mode():
    pe = SinusoidalPositionalEncoding(d_model=8, max_len=16, return_encoding_only=True)
    x = torch.randn(3, 5, 8)
    enc = pe(x)
    assert enc.shape == (3, 5, 8)
