import os
import random
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import LfbDataset, TokenClassificationDataset, collate_fn
from src.models.configs import BertConfig
from src.models.model import (
    BertForSequenceClassification,
    BertForTokenClassification,
    BertMLM,
)
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
            num_labels=self.model_config.get("num_labels", 2),
            tokenizer_name=tokenizer_name,
            tokenizer_padding=self.tokenizer_config.get("padding", "max_length"),
            tokenizer_truncation=self.tokenizer_config.get("truncation", True),
            tokenizer_max_length=self.tokenizer_config.get("max_length", 512),
        )

        if self.task == "mlm":
            self.model = BertMLM(bert_config)
        elif self.task == "classification":
            self.model = BertForSequenceClassification(bert_config)
        elif self.task in ("pos_tagging", "token_classification"):
            self.model = BertForTokenClassification(bert_config)
        else:
            raise ValueError(
                f"Task type '{self.task}' is not currently supported in training pipeline constructor."
            )

        self.model.to(self.device)
        count_parameters(self.model)

        # Mixed precision only helps (and is only valid) on CUDA; GradScaler
        # with enabled=False is a transparent no-op on CPU/MPS.
        self.mixed_precision = (
            self.training_config.get("mixed_precision", False)
            and self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler(
            device="cuda" if self.mixed_precision else "cpu",
            enabled=self.mixed_precision,
        )

        # Optionally resume model weights from a prior checkpoint directory
        # (the one written by save_checkpoint). Optimizer/scheduler/scaler
        # state, if present, is restored later in train() once those objects
        # exist.
        self.resume_from = self.training_config.get("resume_from")
        if self.resume_from:
            model_file = os.path.join(self.resume_from, "model.pt")
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"resume_from model file not found: {model_file}"
                )
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
            logger.info(f"Resumed model weights from {model_file}")

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
                    entity=self.training_config.get("wandb_entity"),
                )
                self.callbacks.append(wandb_cb)
                logger.info("Initialized Weights & Biases Logging Callback.")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize WandB. Proceeding without WandB. Error: {e}"
                )
                self.use_wandb = False

    def _build_task_dataset(self, dataset: Any):
        """Picks the Dataset class matching `self.task`: per-token datasets
        (MLM, sequence classification) use `LfbDataset`; CoNLL-style
        per-token-labeled tasks (POS tagging, NER) need subword-to-word
        label alignment, handled by `TokenClassificationDataset`.
        """
        max_length = self.tokenizer_config.get("max_length", 512)
        if self.task in ("pos_tagging", "token_classification"):
            return TokenClassificationDataset(
                dataset=dataset,
                tokenizer=self.tokenizer,
                max_length=max_length,
                tokens_column=self.project_config.get("tokens_column", "tokens"),
                tags_column=self.project_config.get("tags_column", "pos_tags"),
            )
        return LfbDataset(
            dataset=dataset,
            tokenizer=self.tokenizer,
            max_length=max_length,
            text_column=self.project_config.get("text_column", "text"),
            label_column=self.project_config.get("label_column"),
            mlm=(self.task == "mlm"),
        )

    def setup_dataloaders(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        test_dataset: Optional[Any] = None,
    ) -> None:
        """Prepare train, validation, and test data loaders."""
        batch_size = self.training_config.get("batch_size", 16)

        train_ds = self._build_task_dataset(train_dataset)
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

        self.val_loader = None
        if val_dataset is not None:
            val_ds = self._build_task_dataset(val_dataset)
            self.val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )

        self.test_loader = None
        if test_dataset is not None:
            test_ds = self._build_task_dataset(test_dataset)
            self.test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )

    def train(self) -> Optional[dict]:
        """Main training loop. Returns final test-set metrics, if a test loader was configured."""
        epochs = self.training_config.get("epochs", 3)
        lr = float(self.training_config.get("learning_rate", 5e-5))
        weight_decay = self.training_config.get("weight_decay", 0.01)
        grad_accum_steps = self.training_config.get("gradient_accumulation_steps", 1)
        eval_steps = self.training_config.get("eval_steps", 500)
        save_steps = self.training_config.get("save_steps", 1000)
        checkpoint_dir = self.training_config.get("checkpoint_dir", "./checkpoints")
        max_steps = self.training_config.get("max_steps")

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

        start_epoch = 0
        global_step = 0
        effective_step = 0

        if self.resume_from:
            state_file = os.path.join(self.resume_from, "training_state.pt")
            if os.path.exists(state_file):
                training_state = torch.load(state_file, map_location=self.device)
                optimizer.load_state_dict(training_state["optimizer"])
                scheduler.load_state_dict(training_state["scheduler"])
                self.scaler.load_state_dict(training_state["scaler"])
                start_epoch = training_state.get("epoch", 0)
                global_step = training_state.get("global_step", 0)
                effective_step = training_state.get("effective_step", 0)
                best_val_loss = training_state.get("best_val_loss", float("inf"))
                logger.info(
                    f"Resumed training state from {state_file}: "
                    f"start_epoch={start_epoch}, effective_step={effective_step}. "
                    "Note: resume is epoch-granular, not mid-epoch/batch-granular."
                )
            else:
                logger.warning(
                    f"resume_from set but no training_state.pt found at {state_file}; "
                    "optimizer/scheduler start fresh with the resumed model weights."
                )

        # Counters that must persist across epochs (and across a resumed
        # run) live in one dict, mutated in place by `_train_one_epoch` -
        # simpler than threading four return values through the epoch loop.
        state = {
            "global_step": global_step,
            "effective_step": effective_step,
            "best_val_loss": best_val_loss,
            "stop_training": max_steps is not None and effective_step >= max_steps,
        }
        epoch = max(start_epoch - 1, 0)  # bound even if the loop body never runs

        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(start_epoch, epochs):
            if state["stop_training"]:
                break

            self._train_one_epoch(
                epoch=epoch,
                epochs=epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                grad_accum_steps=grad_accum_steps,
                eval_steps=eval_steps,
                save_steps=save_steps,
                checkpoint_dir=checkpoint_dir,
                max_steps=max_steps,
                save_best=save_best,
                save_last=save_last,
                state=state,
            )

            if state["stop_training"]:
                break

        global_step = state["global_step"]
        effective_step = state["effective_step"]
        best_val_loss = state["best_val_loss"]

        # Final Save
        self.save_checkpoint(
            checkpoint_dir,
            "final",
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            effective_step=effective_step,
            best_val_loss=best_val_loss,
        )
        if save_last:
            self.save_checkpoint(
                checkpoint_dir,
                "last",
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                effective_step=effective_step,
                best_val_loss=best_val_loss,
            )

        # Evaluate on test set if provided
        test_metrics = None
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

        return test_metrics

    def _train_one_epoch(
        self,
        epoch: int,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        criterion: nn.Module,
        grad_accum_steps: int,
        eval_steps: int,
        save_steps: int,
        checkpoint_dir: str,
        max_steps: Optional[int],
        save_best: bool,
        save_last: bool,
        state: dict,
    ) -> None:
        """Runs one training epoch over `self.train_loader`.

        Mutates `state` in place (`global_step`, `effective_step`,
        `best_val_loss`, `stop_training`) rather than returning them, since
        these counters must persist across epochs (and across a resumed
        run) - `train()` reads the final values back out of `state` once
        the epoch loop ends.
        """
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

            # Forward pass (autocast is a no-op when mixed_precision is off)
            with torch.autocast(
                device_type=self.device.type, enabled=self.mixed_precision
            ):
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                # Calculate loss using task criterion
                loss = criterion(logits, labels)
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps

            self.scaler.scale(loss).backward()

            epoch_loss += loss.item() * grad_accum_steps
            state["global_step"] += 1

            if state["global_step"] % grad_accum_steps == 0:
                # Clip gradients to avoid exploding gradients in Transformers
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                state["effective_step"] += 1
                effective_step = state["effective_step"]

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
                    self._run_periodic_evaluation(
                        epoch=epoch,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        checkpoint_dir=checkpoint_dir,
                        save_best=save_best,
                        state=state,
                    )

                # Periodic Checkpointing
                if effective_step % save_steps == 0:
                    self.save_checkpoint(
                        checkpoint_dir,
                        effective_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=state["global_step"],
                        effective_step=effective_step,
                        best_val_loss=state["best_val_loss"],
                    )
                    if save_last:
                        self.save_checkpoint(
                            checkpoint_dir,
                            "last",
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            global_step=state["global_step"],
                            effective_step=effective_step,
                            best_val_loss=state["best_val_loss"],
                        )

                if max_steps is not None and effective_step >= max_steps:
                    logger.info(f"Reached max_steps={max_steps}. Stopping training.")
                    state["stop_training"] = True
                    break

            pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        avg_epoch_loss = epoch_loss / len(self.train_loader)
        logger.info(
            f"Epoch {epoch + 1} finished. Average Train Loss: {avg_epoch_loss:.4f}"
        )

    def _run_periodic_evaluation(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        checkpoint_dir: str,
        save_best: bool,
        state: dict,
    ) -> dict:
        """Runs one validation pass via `evaluate()` (the same reusable
        eval unit used for the final test pass and by `src/__main__.py`'s
        CLI eval mode), logs it, fires callbacks, and checkpoints if it's
        the new best. Mutates `state["best_val_loss"]` in place.
        """
        effective_step = state["effective_step"]
        logger.info(f"Running evaluation at step {effective_step}...")
        eval_metrics = evaluate(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            task=self.task,
        )
        logger.info(f"Eval metrics: {eval_metrics}")

        for cb in self.callbacks:
            cb.on_evaluate(self, step=effective_step, metrics=eval_metrics)

        val_loss = eval_metrics.get("loss", float("inf"))
        if save_best and val_loss < state["best_val_loss"]:
            state["best_val_loss"] = val_loss
            logger.info(
                f"New best validation loss: {val_loss:.4f}. Saving best checkpoint..."
            )
            self.save_checkpoint(
                checkpoint_dir,
                "best",
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=state["global_step"],
                effective_step=effective_step,
                best_val_loss=state["best_val_loss"],
            )

        self.model.train()
        return eval_metrics

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        step_identifier: Union[str, int],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        effective_step: Optional[int] = None,
        best_val_loss: Optional[float] = None,
    ) -> None:
        """Save training state.

        `model.pt` holds only the raw model state_dict, kept backward
        compatible with the eval/test loading path in `src/__main__.py`.
        When `optimizer`/`scheduler` are provided, a sibling
        `training_state.pt` is also written so training can be resumed
        (epoch-granular - see `training.resume_from` in config.yaml) via
        `optimizer.load_state_dict`/`scheduler.load_state_dict`/`scaler.load_state_dict`.
        """
        save_path = os.path.join(
            checkpoint_dir, self.run_name, f"checkpoint-{step_identifier}"
        )
        os.makedirs(save_path, exist_ok=True)

        # Save PyTorch Model weights
        model_file = os.path.join(save_path, "model.pt")
        torch.save(self.model.state_dict(), model_file)

        if optimizer is not None and scheduler is not None:
            training_state = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": (epoch or 0) + 1,  # resume starts at the *next* epoch
                "global_step": global_step or 0,
                "effective_step": effective_step or 0,
                "best_val_loss": best_val_loss if best_val_loss is not None else float("inf"),
            }
            torch.save(training_state, os.path.join(save_path, "training_state.pt"))

        # Save YAML configurations
        write_yaml(self.config, os.path.join(save_path, "config.yaml"))

        logger.info(f"Checkpoint saved successfully to {save_path}")
