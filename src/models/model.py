import torch
from torch import nn

from src.models.configs import BertConfig
from src.models.layers import BertBlock, BertEmbeddings


def init_bert_weights(module: nn.Module, initializer_range: float) -> None:
    """Shared weight-init policy for every BERT head (MLM, classification, ...)."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


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
        self.apply(lambda m: init_bert_weights(m, config.initializer_range))

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


class BertForSequenceClassification(nn.Module):
    """
    BERT model with a sequence classification head, for downstream tasks
    (e.g. GLUE-style single-sentence or sentence-pair classification).

    Pools the [CLS] token (position 0) through a Linear+Tanh, matching the
    standard BERT pooler, then a dropout + linear classifier.

    Args:
        config: BertConfig
            Configuration for the BERT model. `config.num_labels` sets the
            classifier's output dimension.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(lambda m: init_bert_weights(m, config.initializer_range))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        **kwargs,
    ):
        """
        Forward pass for the BERT model with a sequence classification head.

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_labels).
        """
        x = self.bert(input_ids, attention_mask, token_type_ids)
        cls_token = x[:, 0]
        pooled = self.pooler_activation(self.pooler_dense(cls_token))
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class BertForTokenClassification(nn.Module):
    """
    BERT model with a per-token classification head, for tasks like POS
    tagging or NER (CoNLL-2003 style): every position in the sequence gets
    its own label, unlike sequence classification's single pooled [CLS]
    prediction.

    Args:
        config: BertConfig
            Configuration for the BERT model. `config.num_labels` sets the
            classifier's output dimension (e.g. 47 for CoNLL-2003 POS tags).
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(lambda m: init_bert_weights(m, config.initializer_range))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        **kwargs,
    ):
        """
        Forward pass for the BERT model with a per-token classification head.

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, num_labels).
        """
        x = self.bert(input_ids, attention_mask, token_type_ids)
        x = self.dropout(x)
        return self.classifier(x)
