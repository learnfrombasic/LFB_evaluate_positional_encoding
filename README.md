# LFB-Evaluate Positional Encoding

## Overview

This repository builds a BERT-style transformer encoder from scratch in
PyTorch to study one specific question: how much does positional encoding
(PE) choice actually matter, and for what?

It implements 11 PE schemes behind a single config switch, and compares them
across six genuinely different tasks - masked language modeling, length
generalization, single-sentence classification, sentence-*pair* natural
language inference, and two flavors of token classification (POS tagging,
named-entity recognition) - plus a data-augmentation robustness study that
tests *why* the differences appear, not just that they do.

Every run logs metrics to Weights & Biases by default (set `use_wandb:
false` in a config to keep a run fully local); model weights are never
uploaded automatically either way - see "Logging" below.

> [!NOTE]
> Everything here runs on CPU (see `docs/2026-09-10/machine-eval.md`) - the
> host GPU's driver doesn't support the pinned CUDA build. Experiments are
> accordingly small-scale (tiny models, short runs, single seed) and are
> reported throughout as feasibility-scale comparisons, not benchmark
> results - see each experiment's own doc for the honest caveats.

## Key findings

Full write-ups live in `docs/2026-09-10/`; short version:

- **RoPE and Shaw et al.'s relative attention were fixed this session** -
  both were previously applied once at the embedding layer, which silently
  defeats the reason either scheme exists. See `pe-refactor-notes.md` and
  `lessons-rope-relative-bug.md`.
- **Length generalization is structural, not gradual**: absolute/learned-
  style schemes crash outright past their training length; attention-level
  schemes (RoPE, relative, ALiBi, T5-bucketed) keep running with only mild
  degradation. See `length-generalization.md`.
- **A "no positional encoding at all" control changes how every other
  result should be read** - see `nope-baseline-finding.md`. On POS tagging
  specifically, most schemes are statistically indistinguishable from a
  trivial "memorize the word" baseline; only `relative`, `rotary`, and
  `alibi` demonstrably use context.
- **Word-order shuffling proves that claim causally**: destroying word
  order erases `relative`/`rotary`'s advantage entirely (they fall back to
  the no-PE baseline), while a span cutoff that preserves local order barely
  touches it. See `augmentation-robustness-finding.md`.

## Project structure

- `src/models/layers/positional_encoding/` - one file per scheme
  (`sinusoidal.py`, `learnable.py`, `rotary.py`, `relative.py`, `alibi.py`,
  `t5_relative.py`, `kerple.py`, `xpos.py`, `identity.py`), registered in
  `__init__.py`'s `get_pos_encoder`.
- `src/models/model.py` - `BertMLM`, `BertForSequenceClassification`,
  `BertForTokenClassification`: one shared BERT encoder (`BertModel`),
  three task heads. `BertForTokenClassification` serves both POS tagging
  and NER (same code path, different `tags_column`/`num_labels`);
  `BertForSequenceClassification` serves both SST-2 (single sentence) and
  RTE/NLI (sentence pair, via `LfbDataset`'s `text_pair_column`).
- `src/pipelines/` - `train.py` (`Trainer`, with `_train_one_epoch` /
  `_run_periodic_evaluation` as the reusable per-epoch and per-evaluation
  units), `eval.py` (`evaluate()`, the single eval pass shared by training
  and CLI eval mode), `criterions.py`.
- `scripts/` - one driver per experiment: `run_pe_experiments.py` (MLM),
  `run_pe_classification_experiments.py` (SST-2),
  `run_pe_pos_tagging_experiments.py` (CoNLL-2003 POS),
  `run_pe_ner_experiments.py` (CoNLL-2003 NER),
  `run_pe_nli_experiments.py` (GLUE RTE),
  `eval_length_generalization.py` (reuses the MLM checkpoints, no
  retraining), `run_pe_word_shuffle_experiment.py`,
  `run_pe_span_cutoff_experiment.py`.
- `configs/` - `config.yaml` is the reference full-scale config;
  `configs/experiment*.yaml` are the small, CPU-feasible configs the
  scripts above actually use.
- `notebooks/pe_visualizations.ipynb` - visualizes what each scheme
  actually computes: heatmaps, RoPE's rotation-invariance curve, ALiBi's
  per-head bias matrices, real attention weights pulled from trained
  checkpoints via a forward hook, and the NoPE permutation-equivariance
  property made visible.
- `tests/` - one file per scheme, each testing the property that actually
  matters for it (not just shapes), plus cross-cutting regression tests
  covering every registered PE type end-to-end.
- `docs/yyyy-mm-dd/` - dated notes for every stage of work: machine
  evaluation, the correctness fix, each experiment's design and results,
  and the two follow-up findings above.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # tests, notebook execution, plotting
```

## Running

Train with the reference config:

```bash
./run_train.sh config.yaml
```

Reproduce an experiment matrix (each script downloads its own small dataset
once to `data/`, then runs every PE type sequentially, writing results to
`docs/<today>/`):

```bash
python scripts/run_pe_experiments.py                 # MLM comparison
python scripts/run_pe_classification_experiments.py  # SST-2 classification
python scripts/run_pe_pos_tagging_experiments.py      # CoNLL-2003 POS tagging
python scripts/run_pe_ner_experiments.py              # CoNLL-2003 NER
python scripts/run_pe_nli_experiments.py              # GLUE RTE (sentence-pair NLI)
python scripts/eval_length_generalization.py          # train-short-test-long
python scripts/run_pe_word_shuffle_experiment.py      # word-order robustness
python scripts/run_pe_span_cutoff_experiment.py       # span-gap robustness
```

Run the test suite (CPU-only, no network or GPU required):

```bash
pytest tests/ -q
```

## Logging

Every config ships with `use_wandb: true` - training metrics (loss, lr,
eval loss/accuracy) are logged to Weights & Biases per run, in addition to
stdout, via `WandbCallback`. Set `wandb_project`/`wandb_entity` in a config
to control where a given run lands (`wandb_entity: null` defaults to the
authenticated account). Set `use_wandb: false` in a config to keep a run
fully local (stdout + the local checkpoint directory only, nothing
uploaded).

Model *weights* are never uploaded automatically either way: `WandbCallback`
only logs scalar metrics unless you explicitly call its `log_artifact(...)`
method, and `src/callbacks/hf_callback.py`'s `push_to_hub_callback(...)` can
push a full checkpoint folder to the Hugging Face Hub - both exist for when
you want to share/back up a specific checkpoint externally, but neither is
wired into the training loop by default. Checkpoints and datasets always
stay local under `./checkpoints/` and `./data/` (both gitignored)
regardless of the W&B setting.

## Supported positional encodings

Set `model.position_embedding_type` in any config to one of:

| Key | Scheme | Level |
|---|---|---|
| `absolute` / `fixed` / `sinusoidal` | Fixed sinusoidal (Vaswani et al., 2017) | embedding |
| `tape` | tAPE (Foumani et al., 2023) | embedding |
| `learnable` | Learned table, small init | embedding |
| `learned` | Learned table, large init | embedding |
| `rotary` / `rope` | RoPE (Su et al., 2021) | attention |
| `relative` | Shaw et al. (2018) | attention |
| `alibi` | ALiBi (Press et al., 2021) | attention |
| `t5_relative` | T5-style bucketed relative (Raffel et al., 2019) | attention |
| `kerple` | KERPLE, learnable log-distance bias (Chi et al., 2022) | attention |
| `xpos` | xPos, RoPE + relative-distance decay (Sun et al., 2022) | attention |
| `none` / `nope` | No positional encoding (scientific control) | - |

`temporal` (return-encoding-only sinusoidal) is also supported for
completeness; see `src/models/layers/positional_encoding/__init__.py` for
the full registry.

## License

MIT - see `LICENSE`.
