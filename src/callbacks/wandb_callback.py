import wandb


class WandbCallback:
    def __init__(
        self, project_name: str, run_name: str, config: dict = None, entity: str = None
    ):
        """
        Initializes the W&B run.
        """
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            entity=entity,
            reinit=True,
        )
        self.best_accuracy = 0.0

    def on_train_begin(self, trainer) -> None:
        """Called at the start of training.

        Currently logs the start of the run in WandB. Can be extended for custom behavior.
        """
        # Log a simple tag indicating training has begun.
        self.run.log({"train/begin": True})

    def log_metrics(self, metrics, step, prefix="eval"):
        """
        Logs a dictionary of metrics.

        Args:
            metrics (dict): The dictionary from your eval function.
            step (int): Current global step or epoch.
            prefix (str): Dashboard grouping (e.g., 'train' or 'eval').
        """
        # Format keys: {'loss': 0.5} -> {'eval/loss': 0.5}
        log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Log to W&B
        self.run.log(log_dict, step=step)

        # Optional: Track Best Metric Logic
        if "accuracy" in metrics:
            if metrics["accuracy"] > self.best_accuracy:
                self.best_accuracy = metrics["accuracy"]
                self.run.summary["best_accuracy"] = self.best_accuracy
                print(f"New best accuracy: {self.best_accuracy:.4f}")

    def log_artifact(self, model_path, name="model-checkpoint"):
        """Save your model file to W&B"""
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)

    def finish(self):
        """Closes the W&B run"""
        self.run.finish()

    def on_step_end(self, trainer, step: int, loss: float, lr: float) -> None:
        """Called at the end of every training step."""
        self.log_metrics({"loss": loss, "lr": lr}, step=step, prefix="train")

    def on_evaluate(self, trainer, step: int, metrics: dict) -> None:
        """Called at the end of every evaluation run."""
        self.log_metrics(metrics, step=step, prefix="eval")

    def on_train_end(self, trainer) -> None:
        """Called at the end of training."""
        self.finish()
