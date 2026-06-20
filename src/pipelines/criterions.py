import torch
import torch.nn as nn


class MLMCriterion(nn.Module):
    """
    Criterion for Masked Language Modeling (MLM).
    Computes Cross Entropy Loss on 3D logits against 2D labels, ignoring the padding/unmasked tokens.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits shape: (batch_size, sequence_length, vocab_size)
        # labels shape: (batch_size, sequence_length)
        return self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))


class ClassificationCriterion(nn.Module):
    """
    Criterion for Sequence/Token Classification.
    Computes standard Cross Entropy Loss on logits.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits shape: (batch_size, num_classes) or (batch_size * sequence_length, num_classes)
        # labels shape: (batch_size,) or (batch_size * sequence_length,)
        return self.loss_fn(logits, labels)


def get_criterion(task_name: str, **kwargs) -> nn.Module:
    """
    Helper function to instantiate task-specific loss criteria.
    """
    task_name = task_name.lower()
    if task_name == "mlm":
        ignore_index = kwargs.get("ignore_index", -100)
        return MLMCriterion(ignore_index=ignore_index)
    elif task_name == "classification":
        return ClassificationCriterion()
    else:
        raise ValueError(f"Unsupported task type for criterion: {task_name}")
