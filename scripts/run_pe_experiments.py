"""Runs a short comparative feasibility study across every positional
encoding scheme registered in `src/models/layers/positional_encoding`.

Why this exists instead of just looping `run_train.sh`:
  - This machine's GPU (Quadro P2000, 4GB VRAM) can't run the pinned torch
    CUDA build against the installed driver, and 4GB is too little for real
    pretraining anyway - see docs/2026-09-10/machine-eval.md. So this runs
    on CPU, on a small model/subset, capped to a bounded number of steps.
  - `src/__main__.py`'s train mode calls `load_dataset(name, split="train")`,
    which downloads *all* shards of that split (~13 GB here) regardless of
    any later `--subset` slicing. This script instead loads the three local
    parquet shards already downloaded once to `data/{train,validation,test}.parquet`
    (one shard per split - see docs/2026-09-10/experiment-design.md), then
    subsets in memory. Total one-time download for those three shards: ~820MB.

Safety:
  - runs execute strictly sequentially, in-process (never parallel)
  - each run is capped by both `training.max_steps` in the config AND a
    wall-clock SIGALRM timeout, so one stuck run can't block the whole matrix
  - a failing/hanging run is recorded as failed and the script moves on
  - a result note is written immediately after each run finishes, so a
    crash mid-matrix doesn't lose earlier results
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
    sys.path.insert(0, str(ROOT))  # allow `python scripts/run_pe_experiments.py` from anywhere

from datasets import load_dataset  # noqa: E402

from src.pipelines.train import Trainer  # noqa: E402
from src.utils import read_yaml, setup_logger, write_yaml  # noqa: E402

logger = setup_logger("pe_experiments")
BASE_CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
EXPERIMENT_CONFIG_DIR = ROOT / "configs" / "experiments"
DOCS_DIR = ROOT / "docs" / date.today().isoformat()

# The 7 semantically distinct schemes documented in config.yaml (skipping
# aliases that share an implementation: fixed/sinusoidal == absolute,
# rope == rotary) plus the bonus t5_relative type wired up during the
# correctness fix.
PE_TYPES = [
    "absolute",
    "tape",
    "learnable",
    "learned",
    "rotary",
    "relative",
    "alibi",
    "t5_relative",
    "kerple",
    "xpos",
    "none",
]

TRAIN_SUBSET = 2000
VAL_SUBSET = 400
TEST_SUBSET = 400
SPLIT_SEED = 18210
RUN_TIMEOUT_SECONDS = 900  # hard wall-clock ceiling per PE type, on top of max_steps


class RunTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise RunTimeout()


def load_local_splits():
    """Loads the pre-downloaded local shards and takes a small random subset
    of each - never touches the network or the other 44 remote train shards.
    """
    ds = load_dataset(
        "parquet",
        data_files={
            "train": str(ROOT / "data" / "train.parquet"),
            "validation": str(ROOT / "data" / "validation.parquet"),
            "test": str(ROOT / "data" / "test.parquet"),
        },
    )
    train_ds = ds["train"].shuffle(seed=SPLIT_SEED).select(range(TRAIN_SUBSET))
    val_ds = ds["validation"].shuffle(seed=SPLIT_SEED).select(range(VAL_SUBSET))
    test_ds = ds["test"].shuffle(seed=SPLIT_SEED).select(range(TEST_SUBSET))
    return train_ds, val_ds, test_ds


def write_results_note(
    pe_type: str, metrics: dict | None, duration_s: float, error: str | None
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / f"results-{pe_type}.md"
    lines = [f"# Experiment result: `{pe_type}`", ""]
    lines.append(f"- Duration: {duration_s:.1f}s")
    if error:
        lines.append(f"- **Status: FAILED** - {error}")
    elif metrics is None:
        lines.append("- Status: completed, but no test metrics were returned")
    else:
        lines.append("- Status: completed")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {path}")


def run_one(
    pe_type: str, train_ds, val_ds, test_ds
) -> tuple[dict | None, float, str | None]:
    base_config = read_yaml(BASE_CONFIG_PATH)
    config = copy.deepcopy(base_config)
    config["model"]["position_embedding_type"] = pe_type
    config["training"]["checkpoint_dir"] = str(
        ROOT / "checkpoints" / "experiments" / pe_type
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
        trainer.setup_dataloaders(
            train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds
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
    path = DOCS_DIR / "results-summary.md"
    lines = [
        "# PE comparative experiment summary",
        "",
        "Short feasibility-scale comparison, run on CPU with a small model "
        f"(hidden=256, 2 layers), a {TRAIN_SUBSET}-row train subset, and "
        "training.max_steps capped - see machine-eval.md and "
        "experiment-design.md for why. This ranks PE schemes directionally "
        "under this tiny budget; it is not a publication-grade pretraining "
        "result and small gaps between schemes are not conclusive.",
        "",
        "| PE type | loss | perplexity | accuracy | duration (s) | status |",
        "|---|---|---|---|---|---|",
    ]
    for pe_type, metrics, duration, error in rows:
        if error:
            lines.append(f"| {pe_type} | - | - | - | {duration:.1f} | FAILED: {error} |")
        elif metrics is None:
            lines.append(f"| {pe_type} | - | - | - | {duration:.1f} | no metrics |")
        else:
            loss = metrics.get("loss")
            ppl = metrics.get("perplexity")
            acc = metrics.get("accuracy")
            lines.append(
                f"| {pe_type} | {loss:.4f} | {ppl:.2f} | {acc:.4f} | {duration:.1f} | ok |"
            )
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {path}")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds = load_local_splits()
    logger.info(
        f"Loaded subsets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    summary_rows = []
    for pe_type in PE_TYPES:
        logger.info(f"=== Running PE type: {pe_type} ===")
        metrics, duration, error = run_one(pe_type, train_ds, val_ds, test_ds)
        write_results_note(pe_type, metrics, duration, error)
        summary_rows.append((pe_type, metrics, duration, error))

    write_summary(summary_rows)


if __name__ == "__main__":
    main()
