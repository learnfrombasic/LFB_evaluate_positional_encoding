"""Data-augmentation robustness check: train & evaluate POS-tagging with
word order randomly shuffled within every sentence, holding everything else
constant. This complements the NoPE finding (`nope-baseline-finding.md`)
from the opposite direction: instead of removing the positional mechanism,
corrupt what it has to work with, and see which schemes' accuracy actually
depends on real word order versus which are unaffected.

Only word order changes. For each sentence, tokens and their POS tags are
permuted together with the same random permutation - "the"/DT still gets
tagged DT, it's just no longer adjacent to "dog"/NN the way it originally
was. The per-word identity -> tag relationship a NoPE-style model already
relies on (see the per-word-majority baseline in
`pos-tagging-results-summary.md`) is completely preserved; only inter-word
position/adjacency is destroyed. Both train and validation are shuffled, so
this asks "if a scheme only ever sees shuffled sentences, how much of its
clean-data accuracy can it still reach" - a fair like-for-like comparison,
not a train/test mismatch.

Runs a representative subset of PE types rather than the full 8+none, to
keep this a reasonably-scoped follow-up:
  - relative, rotary: the two schemes that clearly beat the per-word
    baseline on clean data - if they're genuinely using word order,
    shuffling should hurt them the most.
  - absolute: an embedding-level scheme that already underperformed the
    per-word baseline on clean data - checks whether shuffling makes an
    already-bad case worse, unchanged, or (surprisingly) better.
  - none: the NoPE control. Shuffling should have ~zero effect, since a
    permutation-equivariant model already cannot use word order at all.
    This is the sanity check on the whole experiment: if `none`'s score
    moves substantially under shuffling, something about the setup (not
    the hypothesis) is suspect.

Safety: identical guardrails to the other runners - sequential, in-process,
SIGALRM-capped, one result note per run written immediately.
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

logger = setup_logger("pe_word_shuffle_experiment")

DOCS_DIR = ROOT / "docs" / date.today().isoformat()
BASE_CONFIG_PATH = ROOT / "configs" / "experiment_pos_tagging.yaml"
EXPERIMENT_CONFIG_DIR = ROOT / "configs" / "experiments_word_shuffle"

PE_TYPES = ["relative", "rotary", "absolute", "none"]

# Clean-data reference accuracy from pos-tagging-results-summary.md, for the
# side-by-side comparison in the written report.
CLEAN_ACCURACY = {
    "relative": 0.867,
    "rotary": 0.864,
    "absolute": 0.787,
    "none": 0.834,
}

TRAIN_SUBSET = 3000
SPLIT_SEED = 18210
RUN_TIMEOUT_SECONDS = 900


class RunTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise RunTimeout()


def _shuffle_example(example: dict, idx: int) -> dict:
    """Permutes tokens and their tags together with the same permutation -
    word identity/tag pairing is preserved, only sentence order is destroyed.
    """
    rng = random.Random(SPLIT_SEED + idx)
    order = list(range(len(example["tokens"])))
    rng.shuffle(order)
    return {
        "tokens": [example["tokens"][i] for i in order],
        "pos_tags": [example["pos_tags"][i] for i in order],
    }


def load_shuffled_splits():
    ds = load_dataset(
        "parquet",
        data_files={
            "train": str(ROOT / "data" / "conll2003_train.parquet"),
            "validation": str(ROOT / "data" / "conll2003_validation.parquet"),
        },
    )
    train_ds = ds["train"].shuffle(seed=SPLIT_SEED).select(range(TRAIN_SUBSET))
    val_ds = ds["validation"]

    train_shuffled = train_ds.map(_shuffle_example, with_indices=True)
    val_shuffled = val_ds.map(_shuffle_example, with_indices=True)
    return train_shuffled, val_shuffled


def write_results_note(
    pe_type: str, metrics: dict | None, duration_s: float, error: str | None
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / f"word-shuffle-results-{pe_type}.md"
    lines = [f"# Word-shuffle augmentation result: `{pe_type}`", ""]
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
            delta = metrics["accuracy"] - clean
            lines.append(f"- clean-data accuracy (reference): {clean:.4f}")
            lines.append(f"- delta vs. clean: {delta:+.4f}")
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {path}")


def run_one(pe_type: str, train_ds, val_ds) -> tuple[dict | None, float, str | None]:
    base_config = read_yaml(BASE_CONFIG_PATH)
    config = copy.deepcopy(base_config)
    config["model"]["position_embedding_type"] = pe_type
    config["training"]["checkpoint_dir"] = str(
        ROOT / "checkpoints" / "experiments_word_shuffle" / pe_type
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
    path = DOCS_DIR / "word-shuffle-results-summary.md"
    lines = [
        "# Word-order shuffle augmentation: does each scheme actually use word order?",
        "",
        "Same CoNLL-2003 POS-tagging setup as `pos-tagging-results-summary.md`, "
        "but every sentence (train and validation) has its words - and their "
        "tags, moved together - randomly permuted before training. Word "
        "identity -> tag information is fully preserved; only real word "
        "order/adjacency is destroyed.",
        "",
        "| PE type | clean accuracy | shuffled accuracy | delta | duration (s) | status |",
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
    train_ds, val_ds = load_shuffled_splits()
    logger.info(f"Loaded shuffled subsets: train={len(train_ds)}, validation={len(val_ds)}")

    summary_rows = []
    for pe_type in PE_TYPES:
        logger.info(f"=== Running PE type: {pe_type} ===")
        metrics, duration, error = run_one(pe_type, train_ds, val_ds)
        write_results_note(pe_type, metrics, duration, error)
        summary_rows.append((pe_type, metrics, duration, error))

    write_summary(summary_rows)


if __name__ == "__main__":
    main()
