# The missing control: what happens with no positional encoding at all

## Why this was added

Four studies had been run (MLM comparison, length generalization, SST-2
classification, CoNLL-2003 POS tagging) before it became clear something
basic was missing: **none of them included a "no positional encoding at
all" condition.** Every conclusion so far was of the form "scheme A beats
scheme B" - there was no zero point. Without knowing what a model with
*zero* positional information does, a claim like "PE choice matters" is
unfalsifiable: any ranking looks meaningful until you check whether the
worst performer is actually still better than nothing.

## What the literature says to expect

A quick literature check ([Haviv et al. 2022](https://arxiv.org/abs/2203.16634),
[Kazemnejad et al. 2023](https://arxiv.org/abs/2305.19466)) draws a sharp
line between *causal* (decoder-only) and *bidirectional* (encoder)
transformers without positional encoding:

- **Causal models** can infer absolute position from the causal attention
  mask alone (a token can count how many tokens precede it), so NoPE causal
  language models are competitive - sometimes *better* at length
  extrapolation than models with an explicit scheme, since there's no
  positional table to run out of.
- **Bidirectional models** (this project's BERT) have no such mask. Full,
  unmasked self-attention with no positional signal is **permutation-
  equivariant**: shuffle the input tokens and every per-token output
  shuffles identically. The literature's expectation, and the standard
  result cited for MLM-style pretraining, is that bidirectional NoPE models
  suffer "significant performance degradation."

`src/models/layers/positional_encoding/identity.py` implements this control
(pure identity - adds nothing, at any layer) and
`tests/models/layers/positional_encoding/test_identity.py` verifies the
permutation-equivariance property directly on the full model (shuffle the
input tokens, un-shuffle the output, assert it matches the unshuffled run
to float precision) - the exact mechanism the literature describes, made
concrete rather than taken on faith.

## What actually happened when it was run

The literature's prediction ("significant degradation") did **not** hold at
this project's scale, and the reason why turned out to be more interesting
than the prediction itself.

| Task | NoPE result | Where it landed |
|---|---|---|
| MLM (`results-summary.md`) | loss 9.057, ppl 8574, acc 25.8% | **Statistically tied with `rotary`** (9.057 / 8574 / 25.8%, matching to 3+ significant figures) |
| SST-2 classification (`classification-results-summary.md`) | 72.5% accuracy | Inside the "escaped the collapse" cluster (`learnable` 72.5%, `rotary` 72.6%) - not with `absolute`/`tape`/`learned` (~51%) |
| POS tagging (`pos-tagging-results-summary.md`) | 83.4% accuracy | Statistically tied with a **trivial per-word-type majority-tag baseline** (83.17%, computed empirically from the training subset) |

None of these are "NoPE wins" - they're all "NoPE ties something," and what
it ties differs by task in a way that's diagnostic rather than surprising
once you look closely:

## Why: each task's ceiling was set by something other than "position vs. no position"

**MLM** is a fill-in-the-blank task over a 15% random mask; a large fraction
of masked tokens are guessable from *which* words are nearby, independent of
exact order or distance. NoPE matching `rotary` here doesn't mean position
is useless for language modeling in general - it means that at this scale
(9.5M parameters, 100 steps, 2,000 examples) neither model had enough
signal to exploit fine-grained positional structure beyond what bag-of-
context already provides. The schemes that *lost* to NoPE (`absolute`,
`tape`, `learned`) didn't lose because they lacked something NoPE has -
they lost because their specific positional signal (large-scale or
non-adaptive) actively hurt training, a separate problem from "is position
useful."

**SST-2 classification** is well documented in the NLP literature as
substantially solvable from bag-of-words lexical cues alone (strong
sentiment words carry most of the signal regardless of position). NoPE
landing exactly inside the successful cluster is consistent with that: a
classifier that reduces to "which sentiment-bearing words are present"
doesn't need position, so removing it entirely costs nothing here. This
also *resolves* an open question from `classification-results-summary.md`:
the original write-up wasn't sure whether the `absolute`/`tape`/`learned`
collapse to the majority-class baseline was about missing positional
information or about a bad optimization trajectory. NoPE having *no*
positional information at all and still succeeding proves it was never
about missing information - it was specifically the large, non-adaptive
perturbation those three schemes inject into the embedding space before the
classifier head has learned anything.

**POS tagging** is the one task where word order plausibly should matter
(local syntax: determiner-before-noun, verb position), and it's the one
where the *initial* framing ("NoPE ties several real schemes!") looked most
surprising - until the right baseline was computed. A **per-word-type
majority tag** baseline (predict whatever tag a word took most often in
training) scores 83.17% on this dataset, because English POS tagging is
largely solvable by word identity alone: "the" is always a determiner, most
words are morphologically unambiguous. NoPE's 83.4% is statistically
indistinguishable from that trivial baseline - which is exactly what a
model with *zero* ability to use context should score, by construction.
Once that baseline is in the picture, the real finding is that only three
schemes (`relative` +3.5pp, `rotary` +3.2pp, `alibi` +1.4pp) demonstrably
beat "memorize the word" - `learnable` and `t5_relative` are statistically
tied with NoPE, and `absolute`/`tape`/`learned` are **worse than doing
nothing**.

## The general lesson

A ranking across positional-encoding schemes, by itself, cannot tell you
whether position matters for a task - it can only tell you which scheme
did best *among the ones tried*. Two baselines were needed to make these
four studies actually load-bearing: a genuine zero-information control
(NoPE), and, for POS tagging specifically, the correct trivial baseline for
*that* task (per-word majority, not global majority). Both were cheap to
add after the fact - NoPE is a one-line identity function, and the
per-word baseline is a handful of lines of Python - and both changed the
conclusions materially. The lesson generalizes past this project: before
reporting "scheme A beats scheme B," check what the cheapest possible
non-scheme (no mechanism, or a lookup table) already achieves on the same
data.

## What this doesn't change

The **length-generalization** finding (`length-generalization.md`) is
untouched by any of this: `absolute`/`tape`/`learnable`/`learned` crash
with a shape error past their training length regardless of how well they
perform within it, while the attention-level schemes (and, trivially, NoPE
itself - an identity function has no length-dependent parameters to run out
of) keep running. That result was never about score comparisons; it's a
structural fact about which schemes have a position-embedding table sized
to a fixed maximum length. Nothing in this document weakens it.
