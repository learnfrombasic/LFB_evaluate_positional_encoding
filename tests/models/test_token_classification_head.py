import torch

from src.models.configs import BertConfig
from src.models.model import BertForTokenClassification


def test_forward_backward_shape():
    config = BertConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=47,
    )
    model = BertForTokenClassification(config)
    input_ids = torch.randint(0, config.vocab_size, (3, 10))
    attention_mask = torch.ones(3, 10, dtype=torch.long)
    token_type_ids = torch.zeros(3, 10, dtype=torch.long)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    assert logits.shape == (3, 10, config.num_labels)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
