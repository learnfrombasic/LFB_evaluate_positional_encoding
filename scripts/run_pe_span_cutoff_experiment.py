"""Data-augmentation robustness check #2: "cutoff" augmentation (Shen et al.,
ConSERT and related contrastive-learning work) - delete one contiguous span
of tokens per sentence (~20% of its length), keeping the remaining tokens in
their original relative order, and see which PE schemes tolerate a gap in
the sequence versus which degrade.

This is a different failure mode from `run_pe_word_shuffle_experiment.py`'s
full reorder: shuffling destroys *all* local order; cutoff keeps every
surviving token's neighbors correct but introduces a discontinuity - a much
closer analogue to truncated context, a redacted sentence, or a dropped
sentence in a longer document. Schemes whose signal is purely a function of
*relative* offset between surviving tokens (relative, rotary, alibi) should
be far less bothered by this than schemes anchored to *absolute* position
(absolute, tape, learned) - a gap shifts every token after it to an absolute
position it never occupied during training, but does not change any
surviving pair's relative offset in the same way full shuffling would.

Same representative PE subset and safety guardrails as the word-shuffle
script - see that file's docstring for the shared rationale.
"""

import copy
import gc
import random
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

logger = setup_logger("pe_span_cutoff_experiment")

DOCS_DIR = ROOT / "docs" / date.today().isoformat()
BASE_CONFIG_PATH = ROOT / "configs" / "experiment_pos_tagging.yaml"
EXPERIMENT_CONFIG_DIR = ROOT / "configs" / "experiments_span_cutoff"

PE_TYPES = ["relative", "rotary", "absolute", "none"]

CLEAN_ACCURACY = {
    "relative": 0.867,
    "rotary": 0.864,
    "absolute": 0.787,
    "none": 0.834,
}

TRAIN_SUBSET = 3000
SPLIT_SEED = 18210
CUTOFF_RATIO = 0.2
RUN_TIMEOUT_SECONDS = 900


class RunTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise RunTimeout()


def _cutoff_example(example: dict, idx: int) -> dict:
    """Deletes one contiguous span (~CUTOFF_RATIO of sentence length),
    keeping the rest in original relative order. No-ops on sentences too
    short to meaningfully cut."""
    rng = random.Random(SPLIT_SEED + idx)
    n = len(example["tokens"])
    if n <= 3:
        return example

    span_len = min(max(1, int(n * CUTOFF_RATIO)), n - 2)
    start = rng.randint(0, n - span_len)
    keep = list(range(0, start)) + list(range(start + span_len, n))
    return {
        "tokens": [example["tokens"][i] for i in keep],
        "pos_tags": [example["pos_tags"][i] for i in keep],
    }


def load_cutoff_splits():
    ds = load_dataset(
        "parquet",
        data_files={
            "train": str(ROOT / "data" / "conll2003_train.parquet"),
            "validation": str(ROOT / "data" / "conll2003_validation.parquet"),
        },
    )
    train_ds = ds["train"].shuffle(seed=SPLIT_SEED).select(range(TRAIN_SUBSET))
    val_ds = ds["validation"]

    train_cutoff = train_ds.map(_cutoff_example, with_indices=True)
    val_cutoff = val_ds.map(_cutoff_example, with_indices=True)
    return train_cutoff, val_cutoff


def write_results_note(
    pe_type: str, metrics: dict | None, duration_s: float, error: str | None
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / f"span-cutoff-results-{pe_type}.md"
    lines = [f"# Span-cutoff augmentation result: `{pe_type}`", ""]
    lines.append(f"- Duration: {duration_s:.1f}s")
    if error:
        lines.append(f"- **Status: FAILED** - {error}")
    elif metrics is None:
        lines.append("- Status: completed, but no metrics were returned")
    else:
        lines.append("- Status: completed")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
        clean = CLEAN_ACCURACY.get(pe_type)
        if clean is not None:
            lines.append(f"- clean-data accuracy (reference): {clean:.4f}")
            lines.append(f"- delta vs. clean: {metrics['accuracy'] - clean:+.4f}")
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {path}")


def run_one(pe_type: str, train_ds, val_ds) -> tuple[dict | None, float, str | None]:
    base_config = read_yaml(BASE_CONFIG_PATH)
    config = copy.deepcopy(base_config)
    config["model"]["position_embedding_type"] = pe_type
    config["training"]["checkpoint_dir"] = str(
        ROOT / "checkpoints" / "experiments_span_cutoff" / pe_type
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
    path = DOCS_DIR / "span-cutoff-results-summary.md"
    lines = [
        "# Span-cutoff augmentation: does each scheme tolerate a gap in the sequence?",
        "",
        f"Same CoNLL-2003 POS-tagging setup, but every sentence (train and "
        f"validation) has one contiguous span (~{int(CUTOFF_RATIO*100)}% of "
        "its length) deleted before training. Unlike full word-shuffling, "
        "surviving tokens keep their original relative order and neighbors - "
        "only a discontinuity (like a redacted phrase, or two originally "
        "non-adjacent sentences stitched together) is introduced.",
        "",
        "| PE type | clean accuracy | cutoff accuracy | delta | duration (s) | status |",
        "|---|---|---|---|---|---|",
    ]
    for pe_type, metrics, duration, error in rows:
        clean = CLEAN_ACCURACY.get(pe_type)
        clean_str = f"{clean * 100:.1f}%" if clean is not None else "-"
        if error:
            lines.append(f"| {pe_type} | {clean_str} | - | - | {duration:.1f} | FAILED: {error} |")
        elif metrics is None:
            lines.append(f"| {pe_type} | {clean_str} | - | - | {duration:.1f} | no metrics |")
        else:
            acc = metrics["accuracy"]
            delta = (acc - clean) * 100 if clean is not None else None
            delta_str = f"{delta:+.1f}pp" if delta is not None else "-"
            lines.append(
                f"| {pe_type} | {clean_str} | {acc * 100:.1f}% | {delta_str} | {duration:.1f} | ok |"
            )
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {path}")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = load_cutoff_splits()
    logger.info(f"Loaded cutoff subsets: train={len(train_ds)}, validation={len(val_ds)}")

    summary_rows = []
    for pe_type in PE_TYPES:
        logger.info(f"=== Running PE type: {pe_type} ===")
        metrics, duration, error = run_one(pe_type, train_ds, val_ds)
        write_results_note(pe_type, metrics, duration, error)
        summary_rows.append((pe_type, metrics, duration, error))

    write_summary(summary_rows)


if __name__ == "__main__":
    main()
