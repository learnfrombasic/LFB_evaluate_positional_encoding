"""Runs the same comparative PE study as scripts/run_pe_experiments.py, but
on a genuinely different task: single-sentence sentiment classification
(GLUE SST-2), instead of masked language modeling.

Why this exists: the MLM comparison and the length-generalization study both
show PE choice matters for language modeling and for handling sequences
longer than training. Neither shows whether it matters for a downstream task
someone would actually deploy. This trains a small `BertForSequenceClassification`
head from scratch (no MLM pretraining reused - a clean, self-contained signal
about PE's effect on supervised classification) per PE type, on the same
seed/subset/step budget throughout, and reports validation accuracy.

Data: `nyu-mll/glue`'s `sst2` config is tiny (~3MB train, <0.1MB validation) -
downloaded in full via hf_hub_download (see data/sst2_{train,validation}.parquet),
no shard-limiting tricks needed (unlike the 15GB MLM dataset). GLUE's official
test split has no public labels, so validation is used for both periodic and
final evaluation, exactly like the local test-shard reuse in the MLM runner.

Safety: same guardrails as scripts/run_pe_experiments.py - runs execute
strictly sequentially, in-process; each run is capped by both
`training.max_steps` and a wall-clock SIGALRM timeout; a failing/hanging run
is recorded and skipped rather than aborting the matrix; a result note is
written immediately after each run finishes.
"""

import copy
import gc
import signal
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

from src.pipelines.train import Trainer  # noqa: E402
from src.utils import read_yaml, setup_logger, write_yaml  # noqa: E402

logger = setup_logger("pe_classification_experiments")

ROOT_DOCS = ROOT / "docs" / date.today().isoformat()
BASE_CONFIG_PATH = ROOT / "configs" / "experiment_classification.yaml"
EXPERIMENT_CONFIG_DIR = ROOT / "configs" / "experiments_classification"

PE_TYPES = [
    "absolute",
    "tape",
    "learnable",
    "learned",
    "rotary",
    "relative",
    "alibi",
    "t5_relative",
]

TRAIN_SUBSET = 3000
SPLIT_SEED = 18210
RUN_TIMEOUT_SECONDS = 900  # hard wall-clock ceiling per PE type, on top of max_steps


class RunTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise RunTimeout()


def load_local_splits():
    """Loads the full local SST-2 parquet files and takes a subset of train.
    Validation (872 rows) is used whole - GLUE's real test split has no
    public labels, so validation stands in for both periodic and final eval.
    """
    ds = load_dataset(
        "parquet",
        data_files={
            "train": str(ROOT / "data" / "sst2_train.parquet"),
            "validation": str(ROOT / "data" / "sst2_validation.parquet"),
        },
    )
    train_ds = ds["train"].shuffle(seed=SPLIT_SEED).select(range(TRAIN_SUBSET))
    val_ds = ds["validation"]
    return train_ds, val_ds


def write_results_note(
    pe_type: str, metrics: dict | None, duration_s: float, error: str | None
) -> None:
    ROOT_DOCS.mkdir(parents=True, exist_ok=True)
    path = ROOT_DOCS / f"classification-results-{pe_type}.md"
    lines = [f"# SST-2 classification result: `{pe_type}`", ""]
    lines.append(f"- Duration: {duration_s:.1f}s")
    if error:
        lines.append(f"- **Status: FAILED** - {error}")
    elif metrics is None:
        lines.append("- Status: completed, but no metrics were returned")
    else:
        lines.append("- Status: completed")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {path}")


def run_one(pe_type: str, train_ds, val_ds) -> tuple[dict | None, float, str | None]:
    base_config = read_yaml(BASE_CONFIG_PATH)
    config = copy.deepcopy(base_config)
    config["model"]["position_embedding_type"] = pe_type
    config["training"]["checkpoint_dir"] = str(
        ROOT / "checkpoints" / "experiments_classification" / pe_type
    )

    EXPERIMENT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = EXPERIMENT_CONFIG_DIR / f"{pe_type}.yaml"
    write_yaml(config, config_path)

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(RUN_TIMEOUT_SECONDS)
    t0 = time.time()
    trainer = None
    try:
        trainer = Trainer(config_path=str(config_path))
        # Validation stands in for both periodic eval and the final "test"
        # metrics `train()` reports - see module docstring.
        trainer.setup_dataloaders(
            train_dataset=train_ds, val_dataset=val_ds, test_dataset=val_ds
        )
        metrics = trainer.train()
        return metrics, time.time() - t0, None
    except RunTimeout:
        return None, time.time() - t0, f"timed out after {RUN_TIMEOUT_SECONDS}s"
    except Exception as e:  # noqa: BLE001 - one PE type failing must not kill the matrix
        logger.exception(f"Run failed for pe_type={pe_type}")
        return None, time.time() - t0, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        del trainer
        gc.collect()


def write_summary(rows: list[tuple[str, dict | None, float, str | None]]) -> None:
    path = ROOT_DOCS / "classification-results-summary.md"
    lines = [
        "# PE comparative experiment summary: SST-2 sentiment classification",
        "",
        "Same 8 PE schemes as the MLM comparison, now fine-tuned from scratch "
        f"(no MLM pretraining reused) on a {TRAIN_SUBSET}-row subset of GLUE "
        "SST-2, evaluated on the full 872-row validation split (GLUE's real "
        "test split has no public labels). Random-guess baseline is 50%.",
        "",
        "| PE type | loss | accuracy | duration (s) | status |",
        "|---|---|---|---|---|",
    ]
    for pe_type, metrics, duration, error in rows:
        if error:
            lines.append(f"| {pe_type} | - | - | {duration:.1f} | FAILED: {error} |")
        elif metrics is None:
            lines.append(f"| {pe_type} | - | - | {duration:.1f} | no metrics |")
        else:
            loss = metrics.get("loss")
            acc = metrics.get("accuracy")
            lines.append(f"| {pe_type} | {loss:.4f} | {acc * 100:.1f}% | {duration:.1f} | ok |")
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {path}")


def main() -> None:
    ROOT_DOCS.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = load_local_splits()
    logger.info(f"Loaded subsets: train={len(train_ds)}, validation={len(val_ds)}")

    summary_rows = []
    for pe_type in PE_TYPES:
        logger.info(f"=== Running PE type: {pe_type} ===")
        metrics, duration, error = run_one(pe_type, train_ds, val_ds)
        write_results_note(pe_type, metrics, duration, error)
        summary_rows.append((pe_type, metrics, duration, error))

    write_summary(summary_rows)


if __name__ == "__main__":
    main()
