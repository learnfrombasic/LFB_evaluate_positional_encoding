from typing import Dict, List

import datasets
import torch
from torch.utils.data import Dataset

from src.models.tokenizer import BertTokenizer


class LfbDataset(Dataset):
    """Supports both MLM pretraining and downstream task fine-tuning."""

    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer: BertTokenizer,
        max_length: int = 512,
        text_column: str = "text",
        label_column: str | None = None,
        mlm: bool = True,
        mlm_probability: float = 0.15,
    ):

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.mlm = mlm
        self.mlm_probability = mlm_probability

        # Validate columns
        if text_column not in dataset.column_names:
            raise ValueError(f"Column '{text_column}' not found")
        if label_column and label_column not in dataset.column_names:
            raise ValueError(f"Column '{label_column}' not found")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        # Tokenize
        encoded = self.tokenizer.encode(
            item[self.text_column],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        output = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

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


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Convert batch dicts to stacked tensors."""
    output = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        # Convert to tensor
        if isinstance(values[0], list):
            output[key] = torch.tensor(values, dtype=torch.long)
        else:
            output[key] = torch.stack(values)

    return output
