import torch

from src.models.configs import BertConfig
from src.models.layers.positional_encoding import NoPositionalEncoding
from src.models.model import BertModel


def test_output_equals_input():
    pe = NoPositionalEncoding()
    x = torch.randn(2, 10, 16)
    assert torch.equal(pe(x), x)


def test_full_model_is_permutation_equivariant():
    """The property NoPE research predicts for *bidirectional* (non-causal)
    encoders like this project's BERT: with no positional signal anywhere,
    full self-attention is permutation-equivariant - shuffle the input
    tokens and every per-token output shuffles identically. Any scheme with
    real positional signal breaks this by design (see the other tests in
    this directory, none of which would pass this check).
    """
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=32,
        position_embedding_type="none",
    )
    model = BertModel(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    token_type_ids = torch.zeros(1, 8, dtype=torch.long)

    perm = torch.randperm(8)
    inv_perm = torch.argsort(perm)

    with torch.no_grad():
        out = model(input_ids, attention_mask, token_type_ids)
        out_on_permuted_input = model(input_ids[:, perm], attention_mask, token_type_ids)

    # Un-permuting the permuted run's output should exactly recover the
    # original run's output - no positional signal means no dependence on
    # *where* in the sequence a token sits, only on its neighbors' content.
    assert torch.allclose(out_on_permuted_input[:, inv_perm], out, atol=1e-4)
