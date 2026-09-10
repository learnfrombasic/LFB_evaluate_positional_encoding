# Data-augmentation robustness: does each scheme actually *use* word order?

## Why this was added

`nope-baseline-finding.md` established, via a zero-information control, that
`relative` and `rotary` are the only two schemes on POS tagging that
demonstrably beat a trivial "memorize the word" baseline (83.17%). That's
evidence they're *capable* of using context - but it's an indirect
argument (they score higher than something that provably can't use
context). The direct version of the same claim is falsifiable: if their
advantage really comes from word order, then *destroying* word order while
keeping every other signal intact should erase that advantage specifically
for those two schemes, and leave schemes that weren't using order alone.

Two complementary augmentations were run, chosen from a literature scan of
PE-relevant (not generic lexical) text augmentation methods - see the
"methods considered" section below.

## Method 1: full word-order shuffle

`scripts/run_pe_word_shuffle_experiment.py`. Every sentence (train and
validation) has its tokens - and their POS tags, moved together - randomly
permuted before training. Word identity -> tag information is fully
preserved (still trivially solvable by "memorize the word"); only real
word order/adjacency is destroyed.

| PE type | clean accuracy | shuffled accuracy | delta |
|---|---|---|---|
| relative | 86.7% | 83.9% | **-2.8pp** |
| rotary | 86.4% | 83.9% | **-2.5pp** |
| absolute | 78.7% | 78.5% | -0.2pp |
| **none (NoPE)** | 83.4% | 83.4% | **-0.0pp** |

This is about as clean a confirmation as a small toy experiment produces:

- **NoPE moves by exactly zero.** A permutation-equivariant model cannot be
  harmed by permuting its input - this is the sanity check on the entire
  experimental design, not just a data point. If NoPE's score had moved,
  something about the augmentation pipeline (not the hypothesis) would be
  suspect.
- **`relative` and `rotary` both collapse to ~83.9%** - landing almost
  exactly on the per-word baseline (83.17%) and the NoPE score (83.4%).
  Their entire advantage over "just memorize the word" (+3.5pp, +3.2pp on
  clean data) came from real word-order information, and shuffling removes
  essentially all of it.
- **`absolute` barely moves** (-0.2pp) because it had little order-derived
  advantage to lose in the first place - on clean data it already
  underperformed the per-word baseline.

## Method 2: span cutoff (a gentler, more realistic corruption)

`scripts/run_pe_span_cutoff_experiment.py`. Instead of destroying all order,
delete one contiguous span (~20% of sentence length) per sentence, keeping
every surviving token's *relative* neighbors intact - closer to truncated
input, a redacted phrase, or a dropped clause than to full shuffling.

| PE type | clean accuracy | cutoff accuracy | delta |
|---|---|---|---|
| relative | 86.7% | 85.5% | -1.2pp |
| rotary | 86.4% | 85.3% | -1.1pp |
| absolute | 78.7% | 77.4% | -1.3pp |
| **none (NoPE)** | 83.4% | 82.6% | -0.8pp |

This is the contrast that makes the shuffle result mean something specific,
rather than "any perturbation hurts relative/rotary most." Here, **all four
schemes drop by roughly the same, small amount (-0.8pp to -1.3pp)** -
including NoPE, which has no positional mechanism at all to be disrupted.
That uniformity is the signal: a gap makes the task marginally harder for
everyone (shorter/discontinuous context, subword-window edge effects near
the cut), but it does **not** disproportionately hurt the schemes that
otherwise rely on relative offset. Cutoff preserves every surviving token's
local neighbors - the exact structure `relative`'s clipped offset table and
`rotary`'s rotation both operate on - so their advantage over baseline
survives largely intact (85.5%/85.3%, both still clearly above the 83.17%
per-word floor), unlike full shuffling, which destroyed that structure and
erased the advantage completely.

**Full shuffle vs. cutoff, side by side:**

| PE type | clean | shuffle (destroys all order) | cutoff (preserves local order) |
|---|---|---|---|
| relative | 86.7% | 83.9% (-2.8pp - advantage gone) | 85.5% (-1.2pp - advantage intact) |
| rotary | 86.4% | 83.9% (-2.5pp - advantage gone) | 85.3% (-1.1pp - advantage intact) |
| absolute | 78.7% | 78.5% (-0.2pp) | 77.4% (-1.3pp) |
| none | 83.4% | 83.4% (-0.0pp) | 82.6% (-0.8pp) |

The pattern is precisely what "these schemes use *relative*, local
structure" predicts: destroy that structure (shuffle) and the advantage
disappears; degrade the input in a way that preserves it (cutoff) and the
advantage mostly survives.

## Methods considered but not run

A broader literature scan (2025 NLP-augmentation surveys; ConSERT and
related contrastive-learning work) surfaces several other candidates:

- **Synonym replacement / random deletion / random insertion (EDA)** -
  standard, well-established, but largely orthogonal to positional
  encoding: they perturb *content*, not *order*, so they wouldn't
  differentiate PE schemes the way shuffle/cutoff do.
- **Back-translation** - too heavy for this project's local/no-GPU
  constraints (needs a second translation model).
- **Adversarial perturbation of the positional embedding itself** (add a
  learned/adversarial delta to the positional signal during training) -
  directly relevant, but a meaningfully larger implementation lift than
  shuffle/cutoff (needs a min-max training loop) - a good candidate for a
  future follow-up rather than this session's scope.
- **Local/adjacent-pair swaps** (gentler than full shuffle - swap only
  nearby tokens) - would sit between shuffle and cutoff in severity;
  flagged as a natural next experiment if finer-grained "how much local
  disorder can each scheme tolerate" resolution is wanted.

## The general lesson

A baseline (NoPE) tells you whether a scheme's *absolute* performance is
distinguishable from using no position at all. An augmentation that
specifically corrupts the signal a mechanism claims to use tells you
whether its *advantage* over that baseline is causally connected to the
thing it claims to model. Both were needed here: the NoPE control
identified which schemes were plausibly using order (`relative`, `rotary`,
`alibi`); shuffling then confirmed the claim directly for the two schemes
tested, by making their advantage disappear exactly when the thing they
were supposedly using was taken away.
