import torch
from datasets import Dataset

from src.dataset import LfbDataset, collate_fn
from src.models.tokenizer import BertTokenizer


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


def _nli_dataset():
    return Dataset.from_dict(
        {
            "premise": ["A dog runs in the park.", "The cat sleeps."],
            "hypothesis": ["An animal is outside.", "The cat is awake."],
            "label": [0, 1],
        }
    )


def test_text_pair_column_produces_real_token_type_ids():
    """NLI-style sentence-pair input: token_type_ids must actually mark the
    second segment with 1s, not silently fall back to the all-zeros default
    every single-sentence task in this project relies on."""
    tokenizer = BertTokenizer(model_id="bert-base-uncased")
    ds = LfbDataset(
        dataset=_nli_dataset(),
        tokenizer=tokenizer,
        max_length=32,
        text_column="premise",
        text_pair_column="hypothesis",
        label_column="label",
        mlm=False,
    )
    item = ds[0]

    assert "token_type_ids" in item
    assert len(item["token_type_ids"]) == 32
    # Both segments must be present: at least one 0 (premise/[CLS]) and at
    # least one 1 (hypothesis) among the real (non-padding) token types.
    assert 0 in item["token_type_ids"]
    assert 1 in item["token_type_ids"]
    assert item["labels"] == 0


def test_missing_text_pair_column_raises():
    tokenizer = BertTokenizer(model_id="bert-base-uncased")
    try:
        LfbDataset(
            dataset=_nli_dataset(),
            tokenizer=tokenizer,
            text_column="premise",
            text_pair_column="nonexistent",
            mlm=False,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing text_pair_column")


def test_collate_fn_handles_token_type_ids_batch():
    tokenizer = BertTokenizer(model_id="bert-base-uncased")
    ds = LfbDataset(
        dataset=_nli_dataset(),
        tokenizer=tokenizer,
        max_length=32,
        text_column="premise",
        text_pair_column="hypothesis",
        label_column="label",
        mlm=False,
    )
    batch = [ds[0], ds[1]]
    out = collate_fn(batch)
    assert out["token_type_ids"].shape == (2, 32)
