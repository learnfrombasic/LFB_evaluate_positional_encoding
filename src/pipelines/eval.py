import math

import torch
import torch.nn as nn
from tqdm import tqdm

from src.pipelines.criterions import get_criterion


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    task: str = "mlm",
) -> dict:
    """
    Evaluates the model on the validation dataset.

    Args:
        model: PyTorch model.
        dataloader: Validation PyTorch DataLoader.
        device: Device to run evaluation on.
        task: Target task ("mlm" or "classification").

    Returns:
        dict: Evaluated metrics (e.g., loss, perplexity, accuracy).
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    # Task specific metrics
    correct_predictions = 0
    total_metric_tokens = (
        0  # For MLM: total masked tokens. For classification: total samples.
    )

    criterion = get_criterion(task)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Extract and move inputs to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Extract token_type_ids if available, otherwise default to zeros
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            else:
                token_type_ids = token_type_ids.to(device)

            # Forward pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # Loss computation using task criterion
            loss = criterion(logits, labels)

            # Metric calculation based on task
            if task == "mlm":
                # Calculate accuracy on masked tokens only
                masked_mask = labels != -100
                predictions = torch.argmax(logits, dim=-1)

                correct = (predictions[masked_mask] == labels[masked_mask]).sum().item()
                total_masked = masked_mask.sum().item()

                correct_predictions += correct
                total_metric_tokens += total_masked

            elif task == "classification":
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == labels).sum().item()

                correct_predictions += correct
                total_metric_tokens += labels.size(0)
            else:
                raise ValueError(f"Unsupported task type for evaluation: {task}")

            # Accumulate metrics
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = (
        correct_predictions / total_metric_tokens if total_metric_tokens > 0 else 0.0
    )

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
    }

    if task == "mlm":
        try:
            metrics["perplexity"] = math.exp(avg_loss)
        except OverflowError:
            metrics["perplexity"] = float("inf")

    return metrics
