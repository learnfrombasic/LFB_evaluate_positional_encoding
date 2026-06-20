import torch
from transformers import BertTokenizerFast


class BertTokenizer:
    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def batch_encode_plus(
        self,
        text: str,
        padding: bool | str = True,
        truncation: bool = True,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.tokenizer.batch_encode_plus(
            text, padding=padding, truncation=truncation, max_length=max_length
        )

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
