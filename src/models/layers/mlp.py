from typing import Optional, Type
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.3,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STTHead(nn.Module):
    """
    Speech-to-Text head for predicting characters or tokens.
    Typically used with CTC Loss, projecting encoder features to the vocabulary space.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 100,  # Number of characters in the vocabulary + 1 for blank token
        drop: float = 0.3,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, in_features)
        Returns:
            Logits of shape (batch_size, seq_len, num_classes)
        """
        x = self.norm(x)
        x = self.drop(x)
        logits = self.fc(x)
        return logits
