# Experiment design

## Scope

A short **comparative feasibility study** across every positional-encoding
scheme, not a publication-grade pretraining run (see `machine-eval.md` for
why: no usable CUDA on this box, and 4GB VRAM would be too little for real
pretraining regardless). Goal: confirm each scheme trains stably end-to-end
post-refactor and get a directional read on relative MLM loss/perplexity
under an identical, tiny compute budget.

## Data

`config.yaml`'s dataset is `8Opt/bert-mlm-experiments-en` (45 train shards +
3 validation/3 test shards, ~15.4 GB total). `src/__main__.py`'s
`load_dataset(name, split="train")` downloads the *entire* named split
regardless of any later `--subset` slicing - confirmed empirically (killed a
`train[:500]` load after it had already pulled 12 of 45 shards, ~3.7 GB, and
cleaned that up). Streaming mode (`streaming=True`) avoids the download but
returns an `IterableDataset`, which isn't compatible with `LfbDataset`'s
map-style `__getitem__` (used for on-the-fly MLM masking).

Resolution: downloaded exactly one shard per split via `hf_hub_download`
directly (bypassing the dataset builder's shard-prefetch behavior) -
`train-00000-of-00045.parquet` (310MB, 1,609,785 rows), one validation shard
(256MB, 1,341,487 rows), one test shard (256MB, 1,341,488 rows). Total: one
time, ~820MB, cached at `data/{train,validation,test}.parquet` (already
covered by `.gitignore`'s `data/*`). `scripts/run_pe_experiments.py` loads
these locally and takes a random (`shuffle(seed=18210)`) subset: 2000 train
/ 400 validation / 400 test rows, identical across every PE type run so the
comparison isolates the PE scheme as the only varying factor.

## Model / training config (`configs/experiment.yaml`)

Reduced from `config.yaml`'s reference full-scale config for CPU feasibility:

| Setting | `config.yaml` (reference) | `configs/experiment.yaml` (this study) |
|---|---|---|
| hidden_size | 768 | 256 |
| num_attention_heads | 2 | 4 (head_size=64, even - required for RoPE) |
| intermediate_size | 3072 | 1024 |
| max_position_embeddings / tokenizer max_length | 512 | 64 |
| batch_size | 8 | 16 |
| epochs | 3 | 1 |
| max_steps (new) | null | 100 |
| mixed_precision (new) | true | true (no-op on CPU here) |

Resulting model: ~9.5M parameters (mostly the `vocab_size=30522 x hidden=256`
embedding table). `training.max_steps=100` bounds every run's optimizer
steps regardless of subset size; `wandb_entity` left `null` so W&B logs to
the authenticated account's default entity (config.yaml's `wandb_entity:
"mlou"` isn't necessarily accessible from this machine's login).

## Runs

`scripts/run_pe_experiments.py` runs these 7 semantically distinct schemes
(skipping aliases `fixed`/`sinusoidal` == `absolute` and `rope` == `rotary`,
which share an implementation) plus the bonus `t5_relative` type wired up
during the correctness fix:

`absolute, tape, learnable, learned, rotary, relative, alibi, t5_relative`

## Safety guardrails

- Runs execute **strictly sequentially, in-process** - never parallel (no
  benefit on CPU, and keeps memory bounded).
- Each run is capped by `training.max_steps=100` **and** a 900s wall-clock
  `SIGALRM` timeout, whichever comes first.
- A failing or timed-out run is caught, logged, and recorded as failed in
  its results note - it does not abort the rest of the matrix.
- A `docs/2026-09-10/results-<pe_type>.md` note is written immediately after
  each run finishes (not batched at the end), so a crash mid-matrix doesn't
  lose earlier results. `results-summary.md` is written last, once all runs
  have completed or failed.
- Smoke-tested first with `max_steps=3` on a 64-row subset before launching
  the full matrix (see terminal transcript this session) - confirmed
  training, periodic eval, best/final/last checkpointing (including the new
  `training_state.pt` resumability), and final test evaluation all work.
