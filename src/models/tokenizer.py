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
        text_pair: str | list[str] | None = None,
        padding: str | bool = "max_length",
        truncation: bool = True,
        max_length: int | None = 512,
        return_tensors: str | None = "pt",
        is_split_into_words: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Encodes a string or list of strings into token IDs and attention masks.

        `is_split_into_words=True` treats `text` as a pre-tokenized list of
        words (e.g. CoNLL-style token classification data) rather than a raw
        string; the returned encoding then exposes `.word_ids()` for
        subword-to-word label alignment.

        `text_pair`, when given, encodes a two-segment input
        (`[CLS] text [SEP] text_pair [SEP]`) - e.g. premise/hypothesis for
        NLI - and the returned `token_type_ids` mark the second segment with
        1s, matching `type_vocab_size=2` in the model config.
        """
        return self.tokenizer(
            text,
            text_pair=text_pair,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=True,
            is_split_into_words=is_split_into_words,
        )

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
