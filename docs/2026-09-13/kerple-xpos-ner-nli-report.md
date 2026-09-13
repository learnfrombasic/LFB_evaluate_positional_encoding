# Session report: two new PE schemes, two new tasks, logging policy

This session extended the project along three axes: added two PE schemes
(KERPLE, xPos), added two tasks (NER, NLI), and clarified the default
logging/artifact-sharing policy. This doc summarizes what changed, why, and
the real experiment results produced for the two new tasks.

## New PE schemes

- **`kerple`** - KERPLE (Chi et al., 2022, https://arxiv.org/abs/2205.09921).
  Generalizes ALiBi: instead of a fixed, hand-picked per-head linear slope,
  it learns a per-head scale `r1` and rate `r2` and applies a logarithmic
  distance kernel, `bias_h(i,j) = -r1_h * log(1 + r2_h * |i-j|)`. Unlike
  ALiBi's bias (cached per sequence length, since it never changes), KERPLE's
  bias is recomputed every forward call because `r1`/`r2` are learned
  parameters. See `src/models/layers/positional_encoding/kerple.py`.
- **`xpos`** - xPos (Sun et al., 2022, https://arxiv.org/abs/2212.10554).
  Extends RoPE with a relative-distance decay: the query at position `n` is
  scaled by `zeta^n`, the key at position `m` by `zeta^-m`, so their dot
  product carries a `zeta^(n-m)` decay factor on top of RoPE's rotation.
  Positions are centered before exponentiation purely for numerical
  stability - this cancels out of every `(n-m)` difference and does not
  change which relative distance the decay depends on (verified directly in
  `tests/.../test_xpos.py`). Caveat noted in the module docstring: the paper
  targets causal (decoder-only) models where `n >= m` always; applied to
  this project's bidirectional attention, the decay is one-directional
  (shrinks scores for `n > m`, grows them for `n < m`) - included as-is, not
  as a claim it's optimal here.

Both are registered in the PE registry (`get_pos_encoder`), wired into
`ScaledDotProductAttention`'s constructor dispatch, covered by dedicated
unit tests (property-focused, not just shape checks), and included in the
cross-cutting `test_full_model_forward_backward_for_every_pe_type`
regression test. Full suite: 55/55 passing.

## New tasks

- **NER** (CoNLL-2003, `ner_tags`, 9 BIO classes) - reused the existing
  `token_classification` code path with zero source changes: same
  `TokenClassificationDataset`, same `BertForTokenClassification` head, just
  a different `tags_column`/`num_labels` in `configs/experiment_ner.yaml`.
  Driver: `scripts/run_pe_ner_experiments.py`.
- **NLI** (GLUE RTE, sentence-pair entailment) - the first task in this
  project to actually need two segments. Required real changes:
  `BertTokenizer.encode` gained a `text_pair` parameter, and `LfbDataset`
  gained a `text_pair_column` option that produces genuine `token_type_ids`
  (every other task here defaults these to all-zeros). `Trainer` passes
  `project.text_pair_column` through automatically. Driver:
  `scripts/run_pe_nli_experiments.py`; config: `configs/experiment_nli.yaml`.
  New tests in `tests/test_dataset.py` cover the sentence-pair path
  directly (real `token_type_ids`, collation, missing-column validation).

Both were smoke-tested end-to-end (tiny step budget) before the full
comparison matrices below were run, to catch pipeline-wiring bugs cheaply.

## Logging / artifact-sharing policy

`use_wandb` now defaults to `true` in all 6 base configs - every run logs
scalar training/eval metrics (loss, lr, eval loss/accuracy) to Weights &
Biases in addition to stdout. Set `use_wandb: false` in a config to keep a
specific run fully local. This is independent of model-weight sharing:
`WandbCallback.log_artifact(...)` (push a checkpoint to W&B) and
`src/callbacks/hf_callback.py`'s `push_to_hub_callback(...)` (push a full
checkpoint folder to the Hugging Face Hub) both still exist for when you
explicitly want to share/back up a checkpoint externally, but neither is
called automatically by the training loop - checkpoints stay local under
`./checkpoints/` unless you call one of these yourself.

The NER and NLI results below were produced with `use_wandb: false` (the
project-wide default at the time these two runs were started) and are
**not** in W&B - only in the local docs/checkpoints. Both configs are now
flipped to `use_wandb: true`, so a future re-run of either script will log
to W&B.

## Results: CoNLL-2003 NER

9 BIO classes, 3000-sentence train subset (same sentences as the
POS-tagging runner), evaluated on the full 3250-sentence validation split.
Full per-scheme detail in `ner-results-<pe_type>.md`; raw table in
`ner-results-summary.md`.

| PE type | loss | accuracy | duration (s) |
|---|---|---|---|
| absolute | 0.4749 | 87.5% | 207.9 |
| tape | 0.4630 | 87.6% | 239.3 |
| learnable | 0.3614 | 91.3% | 243.2 |
| learned | 0.5302 | 86.0% | 237.3 |
| **rotary** | **0.2879** | **93.4%** | 245.1 |
| relative | 0.2959 | 93.3% | 202.7 |
| alibi | 0.3877 | 91.4% | 177.8 |
| t5_relative | 0.3895 | 90.7% | 175.8 |
| kerple | 0.3880 | 91.3% | 173.9 |
| **xpos** | **0.2881** | **93.4%** | 189.7 |
| none | 0.4016 | 90.6% | 200.1 |

Reading this: the same family split this project already established for
POS tagging holds for NER - attention-level schemes cluster on top (rotary
and xpos tie at 93.4%; relative close behind at 93.3%), embedding-level
schemes (absolute, tape, learned) sit at or below the NoPE control (90.6%).
Both new schemes land exactly where their design predicts: xpos tracks
rotary almost exactly (same rotation core, plus a decay term that apparently
doesn't hurt or help much at this scale/length), and kerple lands with
alibi/relative (91.3% vs 91.4%/93.3%), consistent with being a learnable
generalization of ALiBi's bias mechanism. As with every other feasibility-
scale result in this project (single seed, tiny model, short run), read
these as directional, not conclusive - a proper majority-class ('O')
baseline (the same discipline `nope-baseline-finding.md` applied to POS
tagging) has not yet been computed for NER specifically.

## Results: GLUE RTE (NLI)

Sentence-pair entailment, full 2490-row train split, full 277-row
validation split (GLUE's real RTE test split has no public labels).
Random-guess baseline: 50% (2 classes, roughly balanced).

| PE type | loss | accuracy | duration (s) |
|---|---|---|---|
| absolute | 0.7036 | 47.7% | 306.6 |
| tape | 0.7006 | 52.0% | 330.7 |
| **learnable** | 2.2207 | **54.2%** | 447.3 |
| learned | 0.9211 | 47.3% | 531.8 |
| rotary | 2.3185 | 50.5% | 487.8 |
| relative | 2.4634 | 53.4% | 520.3 |
| alibi | 2.3005 | 51.6% | 496.5 |
| t5_relative | 2.3003 | 49.1% | 464.2 |
| kerple | 2.3387 | 50.2% | 476.1 |
| xpos | 2.3237 | 52.7% | 511.0 |
| none | 2.1741 | 52.3% | 557.9 |

Reading this: unlike NER/POS tagging, RTE shows **no clean separation by PE
family** - every scheme lands within a few points of the 50% coin-flip
baseline. This is the weakest of the two new results, and should not be
read as "PE choice doesn't matter for cross-sentence reasoning." RTE is a
genuinely hard task even for large pretrained models; a from-scratch,
9M-parameter model on 2490 rows is very plausibly under-powered for any PE
signal to surface above noise here, consistent with this project's own
established lesson (`nope-baseline-finding.md`) that a task's intrinsic
solvability ceiling has to be accounted for before reading a ranking as
evidence about PE. Rising loss values that don't track accuracy (e.g.
`learnable` has both the best accuracy and a very high loss) further
suggest these small models are overfitting/miscalibrating on a dataset this
small, not learning a clean decision boundary - another reason to treat this
table as a starting point for a follow-up (more epochs, a larger subset
sourced from a different NLI dataset, or multiple seeds) rather than a
finished result.

## What's still open

- Only NER and NLI were run with the new 11-scheme roster. The original
  three comparisons (MLM, SST-2 classification, POS tagging) and the
  length-generalization / word-shuffle / span-cutoff probes have not been
  re-run since `kerple`, `xpos`, and (for the three main scripts) `none`
  were added to their `PE_TYPES` lists - those scripts are ready to produce
  real numbers for the full 11-scheme roster whenever run.
- No majority-class/majority-tag empirical baseline has been computed yet
  for NER (the per-word-majority baseline `nope-baseline-finding.md`
  computed for POS tagging does not directly transfer, since NER's BIO
  scheme is span-based, not just per-word).
- GPU is still unavailable (driver only supports CUDA 12.9; the pinned
  `torch==2.12.0` build needs CUDA 13.0) - see `machine-eval.md`. Not
  attempted this session at the user's request.
