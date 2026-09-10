import torch

from src.models.layers.positional_encoding import RotaryPositionalEncoding


def test_rotate_qk_output_shape():
    rope = RotaryPositionalEncoding(d_model=8, max_len=64)
    q = torch.randn(2, 3, 10, 8)
    k = torch.randn(2, 3, 10, 8)
    q_rot, k_rot = rope.rotate_qk(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_relative_position_invariance():
    """The whole point of RoPE: q_i . k_j after rotation depends only on
    (i - j), not on the absolute positions i, j. Verified purely through the
    public rotate_qk API (no internal cache/method names assumed)."""
    torch.manual_seed(0)
    d = 8
    rope = RotaryPositionalEncoding(d_model=d, max_len=64)
    base_q = torch.randn(1, 1, 1, d)
    base_k = torch.randn(1, 1, 1, d)
    relative_gap = 5

    def score_at(start_pos: int) -> float:
        seq_len = start_pos + relative_gap + 1
        q_full = torch.zeros(1, 1, seq_len, d)
        k_full = torch.zeros(1, 1, seq_len, d)
        q_full[0, 0, start_pos] = base_q
        k_full[0, 0, start_pos + relative_gap] = base_k
        q_rot, k_rot = rope.rotate_qk(q_full, k_full)
        return (q_rot[0, 0, start_pos] @ k_rot[0, 0, start_pos + relative_gap]).item()

    scores = [score_at(start) for start in (0, 3, 10, 20)]
    assert max(scores) - min(scores) < 1e-4


def test_rejects_odd_head_dim():
    try:
        RotaryPositionalEncoding(d_model=7, max_len=16)
    except ValueError:
        return
    raise AssertionError("expected ValueError for odd head dimension")
