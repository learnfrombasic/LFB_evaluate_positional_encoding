import argparse
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.dataset import LfbDataset, collate_fn
from src.models.configs import BertConfig
from src.models.model import BertMLM
from src.models.tokenizer import BertTokenizer
from src.pipelines.eval import evaluate
from src.pipelines.train import Trainer
from src.utils import detect_device, read_yaml, setup_logger

logger = setup_logger("main_entrypoint")


def main():
    parser = argparse.ArgumentParser(
        description="LFB Positional Encoding Pre-training and Evaluation Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Pipeline execution mode",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="If specified, limits the dataset rows (for debugging/dry-runs)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model weights .pt file (required for eval mode)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    config = read_yaml(args.config)
    project_config = config.get("project", {})
    dataset_name = project_config.get("dataset", "8Opt/bert-mlm-experiments-en")
    task = project_config.get("task", "mlm")

    logger.info(f"Execution Mode: {args.mode}")
    logger.info(f"Loading dataset: {dataset_name} for task: {task}...")

    if args.mode == "train":
        # Load dataset splits
        try:
            # Try to load standard train and validation splits
            train_ds = load_dataset(dataset_name, split="train")
            val_ds = load_dataset(dataset_name, split="validation")
        except Exception:
            # Fall back to load full and split manually if validation split is not direct
            full_ds = load_dataset(dataset_name, split="train")
            split_ds = full_ds.train_test_split(test_size=0.1, seed=42)
            train_ds = split_ds["train"]
            val_ds = split_ds["test"]

        if args.subset is not None:
            train_ds = train_ds.select(range(min(len(train_ds), args.subset)))
            val_ds = val_ds.select(
                range(min(len(val_ds), max(1, int(args.subset * 0.2))))
            )
            logger.info(
                f"Sub-sampled dataset: train size = {len(train_ds)}, validation size = {len(val_ds)}"
            )

        trainer = Trainer(config_path=args.config)
        trainer.setup_dataloaders(train_dataset=train_ds, val_dataset=val_ds)

        logger.info("Starting training pipeline...")
        trainer.train()
        logger.info("Training pipeline completed successfully.")

    elif args.mode == "eval":
        if args.model_path is None:
            raise ValueError(
                "A valid --model_path pointing to a .pt file must be specified in eval mode."
            )
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(
                f"Model weight checkpoint file not found: {args.model_path}"
            )

        # Load evaluation split
        try:
            val_ds = load_dataset(dataset_name, split="validation")
        except Exception:
            try:
                val_ds = load_dataset(dataset_name, split="test")
            except Exception:
                full_ds = load_dataset(dataset_name, split="train")
                val_ds = full_ds.train_test_split(test_size=0.1, seed=42)["test"]

        if args.subset is not None:
            val_ds = val_ds.select(range(min(len(val_ds), args.subset)))
            logger.info(f"Sub-sampled evaluation dataset: size = {len(val_ds)}")

        device = detect_device()
        logger.info(f"Using device for evaluation: {device}")

        # Load Config and Initialize Model
        model_config = config.get("model", {})
        tokenizer_config = config.get("tokenizer", {})
        tokenizer_name = tokenizer_config.get("name", "bert-base-uncased")

        bert_config = BertConfig(
            vocab_size=model_config.get("vocab_size", 30522),
            hidden_size=model_config.get("hidden_size", 768),
            num_hidden_layers=model_config.get("num_hidden_layers", 12),
            num_attention_heads=model_config.get("num_attention_heads", 12),
            intermediate_size=model_config.get("intermediate_size", 3072),
            hidden_dropout_prob=model_config.get("hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob=model_config.get(
                "attention_probs_dropout_prob", 0.1
            ),
            max_position_embeddings=model_config.get("max_position_embeddings", 512),
            type_vocab_size=model_config.get("type_vocab_size", 2),
            initializer_range=model_config.get("initializer_range", 0.02),
            layer_norm_eps=float(model_config.get("layer_norm_eps", 1e-12)),
            position_embedding_type=model_config.get(
                "position_embedding_type", "absolute"
            ),
            pad_token_id=model_config.get("pad_token_id", 0),
            pre_layer_norm=model_config.get("pre_layer_norm", False),
            tokenizer_name=tokenizer_name,
            tokenizer_padding=tokenizer_config.get("padding", "max_length"),
            tokenizer_truncation=tokenizer_config.get("truncation", True),
            tokenizer_max_length=tokenizer_config.get("max_length", 512),
        )

        if task == "mlm":
            model = BertMLM(bert_config)
        else:
            raise ValueError(
                f"Task type '{task}' is not currently supported in evaluation pipeline."
            )

        # Load weights
        logger.info(f"Loading weights from {args.model_path}...")
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # Setup Dataloader
        tokenizer = BertTokenizer(model_id=tokenizer_name)
        val_lfb_ds = LfbDataset(
            dataset=val_ds,
            tokenizer=tokenizer,
            max_length=tokenizer_config.get("max_length", 512),
            text_column="text",
            mlm=(task == "mlm"),
        )
        val_loader = DataLoader(
            val_lfb_ds,
            batch_size=config.get("training", {}).get("batch_size", 16),
            shuffle=False,
            collate_fn=collate_fn,
        )

        logger.info("Starting evaluation...")
        eval_metrics = evaluate(
            model=model, dataloader=val_loader, device=device, task=task
        )
        logger.info("Evaluation metrics result:")
        for metric_name, val in eval_metrics.items():
            logger.info(f"  {metric_name}: {val}")


if __name__ == "__main__":
    main()
