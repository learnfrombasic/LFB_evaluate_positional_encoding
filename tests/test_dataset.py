import torch

from src.dataset import collate_fn


def test_collate_fn_handles_scalar_labels():
    """Classification labels are plain ints (not lists), unlike MLM's
    per-token label lists - collate_fn must tensor-ify both correctly."""
    batch = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": 0},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 0], "labels": 1},
    ]
    out = collate_fn(batch)

    assert out["input_ids"].shape == (2, 3)
    assert out["labels"].shape == (2,)
    assert out["labels"].tolist() == [0, 1]


def test_collate_fn_handles_per_token_labels():
    """MLM labels are per-token lists (with -100 for unmasked positions)."""
    batch = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, -100]},
        {"input_ids": [4, 5, 6], "labels": [-100, -100, 6]},
    ]
    out = collate_fn(batch)

    assert out["labels"].shape == (2, 3)
    assert out["labels"][0].tolist() == [-100, 2, -100]


def test_collate_fn_passes_through_tensors():
    batch = [
        {"x": torch.tensor([1, 2])},
        {"x": torch.tensor([3, 4])},
    ]
    out = collate_fn(batch)
    assert out["x"].shape == (2, 2)
