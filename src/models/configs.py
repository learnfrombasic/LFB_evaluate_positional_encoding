from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class BertConfig:
    """
    Configuration class to store the configuration of a `BertModel`.
    It is used to instantiate an BERT model according to the specified arguments,
    defining the model architecture.
    Parameters:
        vocab_size: int
            Vocabulary size of `inputs_ids` in `BertModel`.
        hidden_size: int
            Size of the encoder layers and the pooler layer.
        num_hidden_layers: int
            Number of hidden layers in the Transformer encoder.
        num_attention_heads: int
            Number of attention heads for each attention layer in
            the Transformer encoder.
        intermediate_size: int
            The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        hidden_dropout_prob: float
            The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: float
            The dropout ratio for the attention probabilities.
        max_position_embeddings: int
            The maximum value of the position index.
        type_vocab_size: int
            The vocabulary size of `token_type_ids`.
            The vocabulary size of `token_type_ids`.
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
            0 corresponds to a sentence A token,
            1 corresponds to a sentence B token.
        initializer_range: float
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.

        pre_layer_norm: bool
            Whether to use post layer norm or pre layer norm.
            pre-LN: Layer normalization is applied to the input of the attention block and ffn block, but not to the residual connection.
            post-LN: Layer normalization is applied after the residual connection.

            post-LN is the default in BERT, but pre-LN may improve training stability
            in deep transformer models due to gradient amplification.
            https://arxiv.org/abs/2002.04745
    """

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    position_embedding_type: str = (
        "absolute"  # "absolute", "relative", "rotary", "alibi", etc.
    )
    pad_token_id: int = 0
    pre_layer_norm: bool = False
    tokenizer_name: str = "bert-base-uncased"
    tokenizer_padding: Union[str, bool] = True
    tokenizer_truncation: bool = True
    tokenizer_max_length: Optional[int] = 512
