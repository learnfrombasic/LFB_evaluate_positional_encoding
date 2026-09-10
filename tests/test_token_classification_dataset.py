from datasets import Dataset

from src.dataset import TokenClassificationDataset
from src.models.tokenizer import BertTokenizer


def _tiny_dataset():
    return Dataset.from_dict(
        {
            "tokens": [["EU", "rejects", "boycotting"]],
            "pos_tags": [[22, 42, 37]],
        }
    )


def test_subword_alignment_labels_only_first_subword():
    tokenizer = BertTokenizer(model_id="bert-base-uncased")
    ds = TokenClassificationDataset(
        dataset=_tiny_dataset(),
        tokenizer=tokenizer,
        max_length=16,
        tokens_column="tokens",
        tags_column="pos_tags",
    )
    item = ds[0]

    assert len(item["input_ids"]) == 16
    assert len(item["labels"]) == 16

    # Whatever subwords each word splits into, exactly the words' own tags
    # (in order) must appear among the labels, with everything else -100.
    real_labels = [label for label in item["labels"] if label != -100]
    assert real_labels == [22, 42, 37]

    # [CLS] at position 0 is always ignored.
    assert item["labels"][0] == -100


def test_missing_column_raises():
    tokenizer = BertTokenizer(model_id="bert-base-uncased")
    try:
        TokenClassificationDataset(
            dataset=_tiny_dataset(),
            tokenizer=tokenizer,
            tokens_column="tokens",
            tags_column="nonexistent",
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing tags_column")
