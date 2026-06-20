import torch
from transformers import BertTokenizerFast


class BertTokenizer:
    def __init__(self, model_id: str = "bert-base-uncased"):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_id)

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    def encode(
        self,
        text: str | list[str],
        padding: str | bool = "max_length",
        truncation: bool = True,
        max_length: int | None = 512,
        return_tensors: str | None = "pt",
    ) -> dict[str, torch.Tensor]:
        """Encodes a string or list of strings into token IDs and attention masks."""
        return self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=True,
        )

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
