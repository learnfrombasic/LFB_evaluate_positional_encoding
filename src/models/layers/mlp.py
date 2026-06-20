import torch
from torch import nn

from src.models.layers.attentions import ScaledDotProductAttention


class BertFFN(nn.Module):
    """
    Feed-Forward Network (FFN) used in BERT.

    Args:
        config: BertConfig
            Configuration for the BERT model.
    """

    def __init__(self, config):
        super().__init__()
        self.scale_factor = 4
        self.layers = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * self.scale_factor),
            # approximate GELU with tanh is sufficient for BERT
            nn.GELU(approximate="tanh"),
            nn.Dropout(
                config.hidden_dropout_prob
            ),  # technically, this isn't standard for BERT
            nn.Linear(config.hidden_size * self.scale_factor, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

    def forward(self, x):
        """
        Forward pass for the feed-forward network.

        Args:
            x: torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the feed-forward network.
        """
        return self.layers(x)


class BertBlock(nn.Module):
    """
    Single block of the BERT model, consisting of attention and feed-forward layers.

    Args:
        config: BertConfig
            Configuration for the BERT model.
    """

    def __init__(self, config):
        """
        BERT uses post-LN where layer normalization
        is applied after the residual connection,
        as opposed to pre-LN where layer normalization is applied
        to the input of the attention block and ffn block, but not to the
        residual connection.
        """
        super().__init__()
        self.attention = ScaledDotProductAttention(config)
        self.ffn = BertFFN(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_layer_norm = getattr(config, "pre_layer_norm", False)

    def forward(self, x, attention_mask: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for the BERT block.
        This

        Args:
            x: torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention and feed-forward layers.
        """
        if self.pre_layer_norm:
            x = x + self.attention(self.ln1(x), attention_mask)
        else:
            x = self.ln1(x + self.attention(x, attention_mask))

        if self.pre_layer_norm:
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln2(x + self.ffn(x))
        return x
