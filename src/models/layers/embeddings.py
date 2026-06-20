from typing import Optional

import torch
import torch.nn as nn

from src.models.configs import BertConfig

from .positional_encoding import get_pos_encoder


class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position, and token_type embeddings.

    The default BERT model uses post-LN configuration, where LayerNorm is applied after the residual connection.
    When pre-LN is used, LayerNorm is applied to the input of the attention block and ffn block, but not to the residual connection.
    Because LayerNorm is applied at the input of the attention block, the embedding LayerNorm is redundant.
    The Transformer's first LayerNorm can handle scaling and stabilization directly from the combined embeddings (post-Dropout).

    Args:
        config: BertConfig
            Configuration for the BERT model.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        pos_encoder_cls = get_pos_encoder(self.position_embedding_type)
        self.position_embeddings = pos_encoder_cls(
            d_model=config.hidden_size,
            dropout=0.0,  # TODO: this must be configureable via config.yaml
            max_len=config.max_position_embeddings,
        )

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pre_layer_norm = getattr(config, "pre_layer_norm", False)
        if not self.pre_layer_norm:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the embeddings.

        Args:
            input_ids: torch.Tensor
                Tensor of input token IDs.
            token_type_ids: torch.Tensor
                Tensor of token type IDs.

        Returns:
            torch.Tensor: Combined embeddings.
        """
        T = input_ids.size(1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(T, dtype=torch.long, device=input_ids.device)

        word_emb = self.word_embeddings(input_ids)  # (B,T,768)
        tok_type_emb = self.token_type_embeddings(token_type_ids)  # (B,T,768)
        x = word_emb + tok_type_emb

        pe_out = self.position_embeddings(x)
        if self.position_embedding_type.lower() == "temporal":
            x = x + pe_out
        else:
            x = pe_out

        # LayerNorm stabilizes the combined embeddings by ensuring zero mean and unit variance
        if not self.pre_layer_norm:
            x = self.norm(x)

        # Regularizes the embeddings before they enter the Transformer stack,
        # preventing overfitting to specific embedding patterns.
        x = self.dropout(x)
        return x
