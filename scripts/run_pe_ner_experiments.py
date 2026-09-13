"""Runs the same comparative PE study as the other task runners, on a fourth
structurally different task: named-entity recognition (CoNLL-2003 NER) -
per-token BIO-tagged classification (9 classes: O, B/I-PER, B/I-ORG, B/I-LOC,
B/I-MISC), reusing the exact `tokens`/`ner_tags` columns already present in
the CoNLL-2003 parquet files downloaded for the POS-tagging runner.

Why this exists alongside POS tagging rather than instead of it: POS tagging
is largely solvable from word identity alone (see
docs/2026-09-10/nope-baseline-finding.md - a per-word majority-tag baseline
already gets ~83%). NER is a genuinely different structural test even though
it reuses the same sentences and the same per-token classification head:
entity spans require recognizing *boundaries* (where a B- tag starts vs an
I- tag continues) and often depend on multi-word context (e.g.
disambiguating a capitalized word as a person vs. organization from
surrounding tokens) - a task where position/order plausibly matters more
than for POS tagging's largely local, word-identity-driven decisions.

Data/task plumbing: identical to the POS-tagging runner (same
`TokenClassificationDataset`, same `task: token_classification` code path in
`Trainer` - see `configs/experiment_ner.yaml`), just pointed at `ner_tags`
instead of `pos_tags` and with `num_labels=9` instead of 47. No source
changes were needed to add this task.

Safety: identical guardrails to the other runners - strictly sequential,
in-process; each run capped by `training.max_steps` and a wall-clock SIGALRM
timeout; a failing/hanging run is recorded and skipped; a result note is
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

logger = setup_logger("pe_ner_experiments")

DOCS_DIR = ROOT / "docs" / date.today().isoformat()
BASE_CONFIG_PATH = ROOT / "configs" / "experiment_ner.yaml"
EXPERIMENT_CONFIG_DIR = ROOT / "configs" / "experiments_ner"

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

TRAIN_SUBSET = 3000
SPLIT_SEED = 18210
RUN_TIMEOUT_SECONDS = 900  # hard wall-clock ceiling per PE type, on top of max_steps


class RunTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise RunTimeout()


def load_local_splits():
    """Loads the full local CoNLL-2003 parquet files (already downloaded by
    the POS-tagging runner) and takes a subset of train. The full validation
    split (3250 sentences) is used for both periodic and final evaluation.
    """
    ds = load_dataset(
        "parquet",
        data_files={
            "train": str(ROOT / "data" / "conll2003_train.parquet"),
            "validation": str(ROOT / "data" / "conll2003_validation.parquet"),
        },
    )
    train_ds = ds["train"].shuffle(seed=SPLIT_SEED).select(range(TRAIN_SUBSET))
    val_ds = ds["validation"]
    return train_ds, val_ds


def write_results_note(
    pe_type: str, metrics: dict | None, duration_s: float, error: str | None
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / f"ner-results-{pe_type}.md"
    lines = [f"# NER result: `{pe_type}`", ""]
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
        ROOT / "checkpoints" / "experiments_ner" / pe_type
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
    path = DOCS_DIR / "ner-results-summary.md"
    lines = [
        "# PE comparative experiment summary: CoNLL-2003 NER",
        "",
        f"Same {len(PE_TYPES)} PE schemes as the other task comparisons, trained "
        f"from scratch on a {TRAIN_SUBSET}-sentence subset of CoNLL-2003 (same "
        "sentences as the POS-tagging runner, `ner_tags` instead of "
        "`pos_tags`), evaluated on the full 3250-sentence validation split. "
        "9 BIO classes - a majority-class ('O') baseline should be computed "
        "empirically before reading these numbers as evidence of anything, "
        "the same caveat nope-baseline-finding.md raises for POS tagging.",
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
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
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
