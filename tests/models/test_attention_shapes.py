import pytest
import torch

from src.models.configs import BertConfig
from src.models.model import BertMLM

# Every position_embedding_type value documented as supported in config.yaml,
# plus the bonus t5_relative type.
PE_TYPES = [
    "absolute",
    "fixed",
    "sinusoidal",
    "temporal",
    "tape",
    "learnable",
    "learned",
    "rotary",
    "rope",
    "relative",
    "alibi",
    "t5_relative",
    "none",
    "nope",
]


@pytest.mark.parametrize("pe_type", PE_TYPES)
def test_full_model_forward_backward_for_every_pe_type(pe_type):
    config = BertConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=32,
        position_embedding_type=pe_type,
    )
    model = BertMLM(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    token_type_ids = torch.zeros(2, 10, dtype=torch.long)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    assert logits.shape == (2, 10, config.vocab_size)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
