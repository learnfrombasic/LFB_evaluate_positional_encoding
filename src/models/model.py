import torch
from torch import nn

from src.models.configs import BertConfig
from src.models.layers import BertBlock, BertEmbeddings


class BertModel(nn.Module):
    """
    BERT model ("Bidirectional Encoder Representations from Transformers").

    Args:
        config: BertConfig
            Configuration for the BERT model.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = nn.ModuleList(
            [BertBlock(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor,
    ):
        """
        Forward pass for the BERT model.

        Args:
            input_ids: torch.Tensor
                Tensor of input token IDs.
            attention_mask: torch.Tensor
                Tensor of indices specifying which tokens should be attended to.
            token_type_ids: torch.Tensor
                Tensor of token type IDs.

        Returns:
            torch.Tensor: Output tensor after applying the BERT model.
        """
        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.encoder:
            x = layer(x, attention_mask)
        return x


class BertMLM(nn.Module):
    """
    BERT model with masked language modeling (MLM) head.

    The decision to set bias=False in nn.Linear and manage the bias
    as a separate nn.Parameter stems from weight tying and optimization
    flexibility in BERT's design, particularly during pre-training for Masked Language Modeling (MLM).

    The weight matrix of the MLM head's decoder the nn.Linear mapping (768,vocab_size)
    is tied to the input token embedding matrix (vocab_size,768).
    This allows the model to learn embeddings that are optimized for the MLM task.
    This reduces the number of parameters
    (reusing the embedding weights instead of learning a separate decoder matrix).
    This also Enforces consistency: The same features learned for input tokens are
    used to predict output tokens in MLM.

    Args:
        config: BertConfig
            Configuration for the BERT model.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        pre_layer_norm = getattr(config, "pre_layer_norm", False)
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))
        layers.extend(
            [
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(approximate="tanh"),
            ]
        )
        if not pre_layer_norm:
            layers.append(nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))
        layers.append(nn.Linear(config.hidden_size, config.vocab_size, bias=False))
        self.head = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # weight sharing / weight tying
        self.head[-1].weight = self.bert.embeddings.word_embeddings.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        **kwargs,
    ):
        """
        Forward pass for the BERT model with MLM head.

        Args:
            input_ids: torch.Tensor
                Tensor of input token IDs.
            attention_mask: torch.Tensor
                Tensor of indices specifying which tokens should be attended to.
            token_type_ids: torch.Tensor
                Tensor of token type IDs.

        Returns:
            torch.Tensor: Output tensor after applying the BERT model with MLM head.
        """
        x = self.bert(input_ids, attention_mask, token_type_ids)
        x = self.head(x) + self.bias
        return x
