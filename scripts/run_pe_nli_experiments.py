"""Runs the same comparative PE study as the other task runners, on a fifth
structurally different task: natural language inference (GLUE RTE) -
sentence-*pair* classification (entailment / not_entailment), instead of
MLM, single-sentence classification (SST-2), or per-token classification
(POS tagging / NER).

Why this exists: every other task in this project feeds the model exactly
one segment (`token_type_ids` implicitly all zeros - see `LfbDataset` /
`Trainer`/`evaluate()`'s default-to-zeros fallback). NLI is the first task
here where the model actually needs to relate *two* segments (premise,
hypothesis) separated by a real `[SEP]` and a real `token_type_ids` boundary
(1s over the second segment) - `type_vocab_size=2` in `BertConfig` existed
for exactly this case but was previously unused as long as every task was
single-segment. This tests whether PE choice matters differently when the
model must reason about relative position *across* a segment boundary, not
just within a single contiguous span.

Data: `nyu-mll/glue`'s `rte` config (Recognizing Textual Entailment) - small
(2490 train / 277 validation rows, both tiny relative to SST-2's 67k train
rows), downloaded once and cached locally as
`data/nli_{train,validation}.parquet`, exactly like the SST-2 and
CoNLL-2003 runners. GLUE's official RTE test split has no public labels, so
validation is used for both periodic and final evaluation, same as the
SST-2 runner.

Safety: identical guardrails to the other task runners - strictly
sequential, in-process; each run capped by `training.max_steps` and a
wall-clock SIGALRM timeout; a failing/hanging run is recorded and skipped;
a result note is written immediately after each run finishes.
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

logger = setup_logger("pe_nli_experiments")

DOCS_DIR = ROOT / "docs" / date.today().isoformat()
BASE_CONFIG_PATH = ROOT / "configs" / "experiment_nli.yaml"
EXPERIMENT_CONFIG_DIR = ROOT / "configs" / "experiments_nli"

TRAIN_PARQUET = ROOT / "data" / "nli_train.parquet"
VAL_PARQUET = ROOT / "data" / "nli_validation.parquet"

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

RUN_TIMEOUT_SECONDS = 900  # hard wall-clock ceiling per PE type, on top of max_steps


class RunTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise RunTimeout()


def load_local_splits():
    """Downloads GLUE RTE once to local parquet (if not already present),
    then loads from there - same "download once, reuse locally" pattern as
    the SST-2 and CoNLL-2003 runners.
    """
    if not (TRAIN_PARQUET.exists() and VAL_PARQUET.exists()):
        logger.info("Local NLI parquet files not found - downloading GLUE RTE once.")
        ds = load_dataset("nyu-mll/glue", "rte")
        TRAIN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        ds["train"].to_parquet(str(TRAIN_PARQUET))
        ds["validation"].to_parquet(str(VAL_PARQUET))

    ds = load_dataset(
        "parquet",
        data_files={"train": str(TRAIN_PARQUET), "validation": str(VAL_PARQUET)},
    )
    return ds["train"], ds["validation"]


def write_results_note(
    pe_type: str, metrics: dict | None, duration_s: float, error: str | None
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / f"nli-results-{pe_type}.md"
    lines = [f"# NLI (GLUE RTE) result: `{pe_type}`", ""]
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
        ROOT / "checkpoints" / "experiments_nli" / pe_type
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
        # metrics `train()` reports - GLUE's real test split has no labels.
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
    path = DOCS_DIR / "nli-results-summary.md"
    lines = [
        "# PE comparative experiment summary: GLUE RTE (NLI)",
        "",
        f"Same {len(PE_TYPES)} PE schemes as the other task comparisons, "
        "fine-tuned from scratch (no MLM pretraining reused) on the full "
        "2490-row RTE train split, evaluated on the full 277-row validation "
        "split (GLUE's real test split has no public labels). Random-guess "
        "baseline is 50% (2 classes, roughly balanced).",
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
