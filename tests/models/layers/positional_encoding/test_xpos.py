import torch

from src.models.layers.positional_encoding import XPosPositionalEncoding


def test_rotate_qk_output_shape():
    xpos = XPosPositionalEncoding(d_model=8, max_len=64)
    q = torch.randn(2, 3, 10, 8)
    k = torch.randn(2, 3, 10, 8)
    q_rot, k_rot = xpos.rotate_qk(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_relative_position_invariance_of_rotation():
    """Centering the position exponent at seq_len // 2 must cancel out of
    every (n - m) difference, so - exactly like RoPE - q_i . k_j after
    rotate_qk depends only on the relative gap (i - j), not on the absolute
    positions or the total sequence length."""
    torch.manual_seed(0)
    d = 8
    xpos = XPosPositionalEncoding(d_model=d, max_len=64)
    base_q = torch.randn(1, 1, 1, d)
    base_k = torch.randn(1, 1, 1, d)
    relative_gap = 5

    def score_at(start_pos: int) -> float:
        seq_len = start_pos + relative_gap + 1
        q_full = torch.zeros(1, 1, seq_len, d)
        k_full = torch.zeros(1, 1, seq_len, d)
        q_full[0, 0, start_pos] = base_q
        k_full[0, 0, start_pos + relative_gap] = base_k
        q_rot, k_rot = xpos.rotate_qk(q_full, k_full)
        return (q_rot[0, 0, start_pos] @ k_rot[0, 0, start_pos + relative_gap]).item()

    scores = [score_at(start) for start in (0, 3, 10, 20)]
    assert max(scores) - min(scores) < 1e-3


def test_scale_factor_diverges_from_one_with_distance_from_center():
    """The property RoPE alone doesn't have: the q/k magnitude scale
    (zeta^exponent, exponent centered at seq_len // 2) must move
    monotonically further from 1 as a position gets further from the
    sequence center - this is exactly the mechanism that turns into a
    relative-distance decay in the q.k dot product (see module docstring)."""
    xpos = XPosPositionalEncoding(d_model=8, max_len=256, scale_base=64.0)
    seq_len = 101
    center = seq_len // 2
    scale = xpos._qk_scale(seq_len, torch.device("cpu"), torch.float32)  # (T, d)
    deviation = (scale - 1.0).abs()[:, 0]  # first frequency's deviation from 1

    offsets = [0, 10, 30, 50]
    devs_at_offsets = [deviation[center + o].item() for o in offsets]
    assert devs_at_offsets == sorted(devs_at_offsets)
    assert devs_at_offsets[0] < 1e-6  # at the center, exponent is 0 -> scale == 1


def test_rejects_odd_head_dim():
    try:
        XPosPositionalEncoding(d_model=7, max_len=16)
    except ValueError:
        return
    raise AssertionError("expected ValueError for odd head dimension")
