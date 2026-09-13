from typing import Dict, List

import datasets
import torch
from torch.utils.data import Dataset

from src.models.tokenizer import BertTokenizer


class LfbDataset(Dataset):
    """Supports both MLM pretraining and downstream task fine-tuning.

    `text_pair_column`, when set, encodes each example as a two-segment
    input (`text_column`, `text_pair_column`) - e.g. premise/hypothesis for
    NLI - and includes the tokenizer's `token_type_ids` in the output so the
    model actually sees the segment boundary (`[CLS] premise [SEP]
    hypothesis [SEP]`), rather than the all-zeros default every single-
    sentence task in this project relies on implicitly (see `evaluate()` /
    `Trainer`, which default `token_type_ids` to zeros when absent).
    """

    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer: BertTokenizer,
        max_length: int = 512,
        text_column: str = "text",
        text_pair_column: str | None = None,
        label_column: str | None = None,
        mlm: bool = True,
        mlm_probability: float = 0.15,
    ):

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.text_pair_column = text_pair_column
        self.label_column = label_column
        self.mlm = mlm
        self.mlm_probability = mlm_probability

        # Validate columns
        if text_column not in dataset.column_names:
            raise ValueError(f"Column '{text_column}' not found")
        if text_pair_column and text_pair_column not in dataset.column_names:
            raise ValueError(f"Column '{text_pair_column}' not found")
        if label_column and label_column not in dataset.column_names:
            raise ValueError(f"Column '{label_column}' not found")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        # Tokenize
        encoded = self.tokenizer.encode(
            item[self.text_column],
            text_pair=item[self.text_pair_column] if self.text_pair_column else None,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        output = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if self.text_pair_column:
            output["token_type_ids"] = encoded["token_type_ids"]

        # MLM: mask tokens for pretraining
        if self.mlm:
            output["labels"] = self._create_mlm_labels(output["input_ids"].copy())

        # Downstream: add task labels
        if self.label_column:
            output["labels"] = item[self.label_column]

        return output

    def _create_mlm_labels(self, input_ids: list[int]) -> list[int]:
        """Create MLM labels."""
        labels = [-100] * len(input_ids)
        tokenizer_obj = self.tokenizer.tokenizer

        for idx, token_id in enumerate(input_ids):
            # Skip special tokens
            if token_id in [
                tokenizer_obj.cls_token_id,
                tokenizer_obj.sep_token_id,
                tokenizer_obj.pad_token_id,
            ]:
                continue

            # Mask randomly
            if torch.rand(1).item() < self.mlm_probability:
                input_ids[idx] = tokenizer_obj.mask_token_id
                labels[idx] = token_id

        return labels


class TokenClassificationDataset(Dataset):
    """Per-token tasks (POS tagging, NER, ...): input is pre-tokenized into
    words with one label per word (CoNLL-2003 style). Subword tokenization
    splits words into multiple pieces, so labels must be realigned: each
    word's label is assigned only to its first subword; special tokens
    ([CLS]/[SEP]/[PAD]) and continuation subwords get -100 (ignored by the
    loss), following the standard HuggingFace token-classification recipe.
    """

    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer: BertTokenizer,
        max_length: int = 128,
        tokens_column: str = "tokens",
        tags_column: str = "pos_tags",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokens_column = tokens_column
        self.tags_column = tags_column

        if tokens_column not in dataset.column_names:
            raise ValueError(f"Column '{tokens_column}' not found")
        if tags_column not in dataset.column_names:
            raise ValueError(f"Column '{tags_column}' not found")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        words = item[self.tokens_column]
        tags = item[self.tags_column]

        encoded = self.tokenizer.encode(
            words,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
            is_split_into_words=True,
        )

        word_ids = encoded.word_ids()
        labels = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None or word_id == previous_word_id:
                labels.append(-100)
            else:
                labels.append(tags[word_id])
            previous_word_id = word_id

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Convert batch dicts to stacked tensors.

    Handles both per-token fields (input_ids, attention_mask, MLM labels -
    lists of ints) and per-example scalar fields (classification labels -
    plain ints), since `torch.stack` only accepts already-built tensors.
    """
    output = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if isinstance(values[0], torch.Tensor):
            output[key] = torch.stack(values)
        else:
            output[key] = torch.tensor(values, dtype=torch.long)

    return output
