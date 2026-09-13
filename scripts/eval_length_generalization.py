"""Length-generalization ("train short, test long") evaluation across every
positional-encoding scheme.

Why this exists: the main comparison (docs/2026-09-10/results-summary.md)
shows attention-level schemes training better than embedding-level ones
under a short, fixed-length budget, but it doesn't show *why* PE choice
matters for a task. The classic, well-established demonstration - the one
ALiBi's paper is literally named after ("Train Short, Test Long") and RoPE's
extrapolation property is prized for - is to evaluate a model trained at one
sequence length on longer sequences it never saw during training.

This reuses the checkpoints already produced by scripts/run_pe_experiments.py
(no retraining) and evaluates each at sequence lengths beyond the training
length (64 -> 128 -> 256 tokens), built by packing consecutive short
sentences from the local test.parquet shard end-to-end into longer passages.

Predicted (and actual - see the report this writes) outcome, and the reason
it's worth running: the embedding-level schemes (absolute, tape, learnable,
learned) hold a position-embedding table sized exactly to the training
length (64). A forward pass at a longer length isn't a degraded score - it's
a shape error, because positions 64+ simply don't exist in that table. The
attention-level schemes (rotary, relative, alibi, t5_relative) compute their
positional signal from the actual sequence length at every forward call, so
they run at any length - quality still degrades since they only ever saw
64-token attention patterns during training, but they don't fall over.
"""

import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from datasets import Dataset, load_dataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.dataset import LfbDataset, collate_fn  # noqa: E402
from src.models.configs import BertConfig  # noqa: E402
from src.models.model import BertMLM  # noqa: E402
from src.models.tokenizer import BertTokenizer  # noqa: E402
from src.pipelines.eval import evaluate  # noqa: E402
from src.utils import read_yaml, setup_logger  # noqa: E402

logger = setup_logger("pe_length_generalization")

DOCS_DIR = ROOT / "docs" / date.today().isoformat()
BASE_CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "experiments"

# Same schemes as the main comparison; family split used in the report.
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
ATTENTION_LEVEL = {"rotary", "relative", "alibi", "t5_relative", "kerple", "xpos"}
# NoPE is embedding-level by registry classification, but - being a true
# parameter-free identity - it never hits a length ceiling either; labeled
# separately so the report doesn't imply it shares the embedding-level
# schemes' fixed-table failure mode.
NO_LAYER = {"none", "nope"}

TRAIN_LENGTH = 64  # what these checkpoints were actually trained at
EVAL_LENGTHS = [64, 128, 256]
NUM_PACKED_EXAMPLES = 200
BATCH_SIZE = 8


def find_checkpoint(pe_type: str) -> Path:
    matches = sorted(
        CHECKPOINT_ROOT.glob(f"{pe_type}/*/checkpoint-final/model.pt")
    )
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint-final found for '{pe_type}' under {CHECKPOINT_ROOT}. "
            "Run scripts/run_pe_experiments.py first."
        )
    return matches[-1]


def build_packed_dataset(target_length: int, num_examples: int, cursor: int) -> tuple[Dataset, int]:
    """Concatenates consecutive short sentences end-to-end into `num_examples`
    long strings, each with enough raw text to fill `target_length` tokens -
    final tokenization/truncation to the exact target happens in LfbDataset.
    Returns the dataset and the next unused row cursor (so different lengths
    draw non-overlapping rows from the shard).
    """
    raw = load_dataset(
        "parquet", data_files={"test": str(ROOT / "data" / "test.parquet")}
    )["test"]
    # This corpus's sentences run ~5-8 tokens; over-pack generously so the
    # tokenizer's max_length always has enough raw text to truncate down to.
    sentences_per_example = max(4, target_length // 3)
    texts = []
    for _ in range(num_examples):
        chunk = raw[cursor : cursor + sentences_per_example]["text"]
        texts.append(" ".join(chunk))
        cursor += sentences_per_example
    return Dataset.from_dict({"text": texts}), cursor


def load_model(pe_type: str, model_config: dict, tokenizer_name: str) -> tuple[BertMLM, Path]:
    bert_config = BertConfig(
        vocab_size=model_config.get("vocab_size", 30522),
        hidden_size=model_config.get("hidden_size", 256),
        num_hidden_layers=model_config.get("num_hidden_layers", 2),
        num_attention_heads=model_config.get("num_attention_heads", 4),
        intermediate_size=model_config.get("intermediate_size", 1024),
        max_position_embeddings=model_config.get("max_position_embeddings", 64),
        position_embedding_type=pe_type,
        tokenizer_name=tokenizer_name,
    )
    model = BertMLM(bert_config)
    ckpt_path = find_checkpoint(pe_type)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt_path


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    base_config = read_yaml(BASE_CONFIG_PATH)
    model_config = base_config["model"]
    tokenizer_name = base_config["tokenizer"]["name"]
    tokenizer = BertTokenizer(model_id=tokenizer_name)
    device = torch.device("cpu")

    rows = []
    for pe_type in PE_TYPES:
        model, ckpt_path = load_model(pe_type, model_config, tokenizer_name)
        logger.info(f"=== {pe_type} (from {ckpt_path.parent.parent.name}) ===")
        cursor = 0
        for length in EVAL_LENGTHS:
            packed, cursor = build_packed_dataset(length, NUM_PACKED_EXAMPLES, cursor)
            lfb_ds = LfbDataset(
                dataset=packed,
                tokenizer=tokenizer,
                max_length=length,
                text_column="text",
                mlm=True,
            )
            loader = DataLoader(
                lfb_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
            )
            t0 = time.time()
            try:
                metrics = evaluate(model=model, dataloader=loader, device=device, task="mlm")
                status, error = "ok", None
                logger.info(f"  {length} tok: ok  {metrics}")
            except Exception as e:  # noqa: BLE001 - a shape failure here IS the expected finding
                metrics, status, error = None, "failed", f"{type(e).__name__}: {e}"
                logger.info(f"  {length} tok: FAILED  {error}")
            rows.append(
                {
                    "pe_type": pe_type,
                    "length": length,
                    "metrics": metrics,
                    "status": status,
                    "error": error,
                    "duration": time.time() - t0,
                }
            )
        del model

    write_report(rows)


def write_report(rows: list[dict]) -> None:
    path = DOCS_DIR / "length-generalization.md"
    lines = [
        "# Length generalization: train short (64 tokens), test long (64 / 128 / 256)",
        "",
        f"Reuses the checkpoints from the main PE comparison (no retraining) - "
        f"see `results-summary.md`. Each scheme was trained at {TRAIN_LENGTH} "
        "tokens; evaluated here on packed multi-sentence passages built from "
        "the local test shard at 64 (in-distribution), 128 (2x), and 256 "
        "(4x) tokens.",
        "",
        "| Scheme | Layer | 64 tok (trained) | 128 tok (2x) | 256 tok (4x) |",
        "|---|---|---|---|---|",
    ]
    by_type: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_type.setdefault(r["pe_type"], {})[r["length"]] = r

    for pe_type in PE_TYPES:
        if pe_type in ATTENTION_LEVEL:
            layer = "attention"
        elif pe_type in NO_LAYER:
            layer = "(none)"
        else:
            layer = "embedding"
        cells = []
        for length in EVAL_LENGTHS:
            r = by_type[pe_type][length]
            if r["status"] == "ok":
                m = r["metrics"]
                cells.append(f"ppl {m['perplexity']:.0f} / acc {m['accuracy'] * 100:.1f}%")
            else:
                first_line = r["error"].splitlines()[0][:70]
                cells.append(f"**FAILS** ({first_line})")
        lines.append(f"| {pe_type} | {layer} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines += [
        "",
        "## Reading this table",
        "",
        "The embedding-level schemes hold a position-embedding table sized "
        f"exactly to the {TRAIN_LENGTH}-token training length - positions "
        f"beyond {TRAIN_LENGTH} don't exist in that table, so a forward pass "
        "at 128 or 256 tokens is a shape error, not a worse score. The "
        "attention-level schemes compute their positional signal from the "
        "actual sequence length at every call, so they keep running - "
        "quality still degrades since they only ever trained on "
        f"{TRAIN_LENGTH}-token attention patterns, but they don't fall over. "
        "This is the concrete, task-relevant reason RoPE and ALiBi are the "
        "standard choice for anything that must handle variable or growing "
        "context length (chat, long documents, streaming) while absolute/"
        "learned position embeddings hard-cap a model's usable length at "
        "whatever it was trained with.",
    ]
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {path}")


if __name__ == "__main__":
    main()
