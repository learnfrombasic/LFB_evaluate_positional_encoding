import os
import random
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import LfbDataset, collate_fn
from src.models.configs import BertConfig
from src.models.model import BertMLM
from src.models.tokenizer import BertTokenizer
from src.pipelines.criterions import get_criterion
from src.pipelines.eval import evaluate
from src.utils import (
    count_parameters,
    detect_device,
    get_run_name,
    read_yaml,
    setup_logger,
    write_yaml,
)

logger = setup_logger("train_pipeline")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Callbacks are duck-typed and import hooks directly from src/callbacks


class Trainer:
    def __init__(self, config_path: str) -> None:
        self.config = read_yaml(config_path)
        self.device = detect_device()
        logger.info(f"Using device: {self.device}")

        # Extract project configs
        self.project_config = self.config.get("project", {})
        self.model_config = self.config.get("model", {})
        self.tokenizer_config = self.config.get("tokenizer", {})
        self.training_config = self.config.get("training", {})

        # Setup run properties
        self.task = self.project_config.get("task", "mlm")
        self.seed = self.training_config.get("seed", 42)
        set_seed(self.seed)

        self.run_name = get_run_name(
            model_name=self.model_config.get("position_embedding_type", "absolute"),
            dset_name=self.project_config.get("dataset", "dataset").split("/")[-1],
            lr=self.training_config.get("learning_rate", 5e-5),
            batch_size=self.training_config.get("batch_size", 16),
            experiment_type=self.task,
            random_suffix=True,
        )
        logger.info(f"Run name: {self.run_name}")

        # Initialize Tokenizer
        tokenizer_name = self.tokenizer_config.get("name", "bert-base-uncased")
        self.tokenizer = BertTokenizer(model_id=tokenizer_name)

        # Build Model
        bert_config = BertConfig(
            vocab_size=self.model_config.get("vocab_size", 30522),
            hidden_size=self.model_config.get("hidden_size", 768),
            num_hidden_layers=self.model_config.get("num_hidden_layers", 12),
            num_attention_heads=self.model_config.get("num_attention_heads", 12),
            intermediate_size=self.model_config.get("intermediate_size", 3072),
            hidden_dropout_prob=self.model_config.get("hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob=self.model_config.get(
                "attention_probs_dropout_prob", 0.1
            ),
            max_position_embeddings=self.model_config.get(
                "max_position_embeddings", 512
            ),
            type_vocab_size=self.model_config.get("type_vocab_size", 2),
            initializer_range=self.model_config.get("initializer_range", 0.02),
            layer_norm_eps=float(self.model_config.get("layer_norm_eps", 1e-12)),
            position_embedding_type=self.model_config.get(
                "position_embedding_type", "absolute"
            ),
            pad_token_id=self.model_config.get("pad_token_id", 0),
            pre_layer_norm=self.model_config.get("pre_layer_norm", False),
            tokenizer_name=tokenizer_name,
            tokenizer_padding=self.tokenizer_config.get("padding", "max_length"),
            tokenizer_truncation=self.tokenizer_config.get("truncation", True),
            tokenizer_max_length=self.tokenizer_config.get("max_length", 512),
        )

        if self.task == "mlm":
            self.model = BertMLM(bert_config)
        else:
            raise ValueError(
                f"Task type '{self.task}' is not currently supported in training pipeline constructor."
            )

        self.model.to(self.device)
        count_parameters(self.model)

        self.callbacks: list[Any] = []

        # Setup WandB callback
        self.use_wandb = self.training_config.get("use_wandb", False)
        if self.use_wandb:
            try:
                from src.callbacks.wandb_callback import WandbCallback

                wandb_cb = WandbCallback(
                    project_name=self.training_config.get(
                        "wandb_project", "LFB-PE-Eval"
                    ),
                    run_name=self.run_name,
                    config=self.config,
                )
                self.callbacks.append(wandb_cb)
                logger.info("Initialized Weights & Biases Logging Callback.")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize WandB. Proceeding without WandB. Error: {e}"
                )
                self.use_wandb = False

    def setup_dataloaders(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        test_dataset: Optional[Any] = None,
    ) -> None:
        """Prepare train, validation, and test data loaders."""
        batch_size = self.training_config.get("batch_size", 16)

        train_lfb_ds = LfbDataset(
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            max_length=self.tokenizer_config.get("max_length", 512),
            text_column="text",
            mlm=(self.task == "mlm"),
        )

        self.train_loader = DataLoader(
            train_lfb_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

        self.val_loader = None
        if val_dataset is not None:
            val_lfb_ds = LfbDataset(
                dataset=val_dataset,
                tokenizer=self.tokenizer,
                max_length=self.tokenizer_config.get("max_length", 512),
                text_column="text",
                mlm=(self.task == "mlm"),
            )
            self.val_loader = DataLoader(
                val_lfb_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )

        self.test_loader = None
        if test_dataset is not None:
            test_lfb_ds = LfbDataset(
                dataset=test_dataset,
                tokenizer=self.tokenizer,
                max_length=self.tokenizer_config.get("max_length", 512),
                text_column="text",
                mlm=(self.task == "mlm"),
            )
            self.test_loader = DataLoader(
                test_lfb_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )

    def train(self) -> None:
        """Main training loop."""
        epochs = self.training_config.get("epochs", 3)
        lr = float(self.training_config.get("learning_rate", 5e-5))
        weight_decay = self.training_config.get("weight_decay", 0.01)
        grad_accum_steps = self.training_config.get("gradient_accumulation_steps", 1)
        eval_steps = self.training_config.get("eval_steps", 500)
        save_steps = self.training_config.get("save_steps", 1000)
        checkpoint_dir = self.training_config.get("checkpoint_dir", "./checkpoints")

        # Setup optimizer with weight decay exceptions (no decay for bias, norms, embeddings)
        decay_params = []
        nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                "bias" in name
                or "LayerNorm" in name
                or "norm" in name
                or "embeddings" in name
            ):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr)

        # Compute steps
        total_steps = len(self.train_loader) * epochs
        effective_total_steps = total_steps // grad_accum_steps
        warmup_steps = int(
            effective_total_steps * self.training_config.get("warmup_ratio", 0.1)
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=effective_total_steps,
        )

        logger.info(f"Total Epochs: {epochs}")
        logger.info(f"Total Steps: {total_steps}")
        logger.info(f"Effective Steps (accumulated): {effective_total_steps}")
        logger.info(f"Warmup Steps: {warmup_steps}")

        criterion = get_criterion(self.task)
        # Checkpoint flags
        save_best = self.training_config.get("save_best", True)
        save_last = self.training_config.get("save_last", True)
        best_val_loss = float("inf")

        global_step = 0
        effective_step = 0

        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Fetch token_type_ids or default to zeros
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is None:
                    token_type_ids = torch.zeros_like(input_ids)
                else:
                    token_type_ids = token_type_ids.to(self.device)

                # Forward pass
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

                # Calculate loss using task criterion
                loss = criterion(logits, labels)

                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
                loss.backward()

                epoch_loss += loss.item() * grad_accum_steps
                global_step += 1

                if global_step % grad_accum_steps == 0:
                    # Clip gradients to avoid exploding gradients in Transformers
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    effective_step += 1

                    # Run callbacks step end
                    for cb in self.callbacks:
                        cb.on_step_end(
                            self,
                            step=effective_step,
                            loss=loss.item() * grad_accum_steps,
                            lr=scheduler.get_last_lr()[0],
                        )

                    # Periodic Evaluation
                    if effective_step % eval_steps == 0 and self.val_loader is not None:
                        logger.info(f"Running evaluation at step {effective_step}...")
                        eval_metrics = evaluate(
                            model=self.model,
                            dataloader=self.val_loader,
                            device=self.device,
                            task=self.task,
                        )
                        logger.info(f"Eval metrics: {eval_metrics}")

                        # Run callbacks eval
                        for cb in self.callbacks:
                            cb.on_evaluate(
                                self, step=effective_step, metrics=eval_metrics
                            )

                        # Track best model checkpoint
                        val_loss = eval_metrics.get("loss", float("inf"))
                        if save_best and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            logger.info(
                                f"New best validation loss: {best_val_loss:.4f}. Saving best checkpoint..."
                            )
                            self.save_checkpoint(checkpoint_dir, "best")

                        self.model.train()

                    # Periodic Checkpointing
                    if effective_step % save_steps == 0:
                        self.save_checkpoint(checkpoint_dir, effective_step)
                        if save_last:
                            self.save_checkpoint(checkpoint_dir, "last")

                pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            logger.info(
                f"Epoch {epoch + 1} finished. Average Train Loss: {avg_epoch_loss:.4f}"
            )

        # Final Save
        self.save_checkpoint(checkpoint_dir, "final")
        if save_last:
            self.save_checkpoint(checkpoint_dir, "last")

        # Evaluate on test set if provided
        if self.test_loader is not None:
            logger.info("Running evaluation on test set...")
            test_metrics = evaluate(
                model=self.model,
                dataloader=self.test_loader,
                device=self.device,
                task=self.task,
            )
            logger.info(f"Final test metrics: {test_metrics}")

            # Run callbacks eval for test
            for cb in self.callbacks:
                cb.on_evaluate(self, step=effective_step, metrics=test_metrics)

        # Run callbacks train end
        for cb in self.callbacks:
            cb.on_train_end(self)

    def save_checkpoint(
        self, checkpoint_dir: str, step_identifier: Union[str, int]
    ) -> None:
        """Save training states."""
        save_path = os.path.join(
            checkpoint_dir, self.run_name, f"checkpoint-{step_identifier}"
        )
        os.makedirs(save_path, exist_ok=True)

        # Save PyTorch Model weights
        model_file = os.path.join(save_path, "model.pt")
        torch.save(self.model.state_dict(), model_file)

        # Save YAML configurations
        write_yaml(self.config, os.path.join(save_path, "config.yaml"))

        logger.info(f"Checkpoint saved successfully to {save_path}")
