# PE comparative experiment summary: CoNLL-2003 POS tagging

Same 8 PE schemes as the MLM and SST-2 comparisons, trained from scratch on a 3000-sentence subset of CoNLL-2003, evaluated on the full 3250-sentence validation split.

**Two baselines, not one - the second one matters more.** A global majority-tag baseline (always predict the single most common POS tag) measures ~16.5% - a weak reference, since there are 47 classes. The baseline that actually matters for interpreting these numbers is the **per-word-type majority tag** (for each word, predict whatever tag it took most often in training; fall back to the global majority for unseen words) - a standard, trivial "just memorize the word" POS tagger, measured empirically at **83.17%** on this validation split (14.5% of validation tokens are unseen word types). English POS tagging is largely solvable this way because most words are morphologically or lexically unambiguous ("the" is always a determiner); the genuinely hard ~15-20% of tokens are the ones whose tag depends on context (e.g. "book" as noun vs. verb).

| PE type | loss | accuracy | vs. per-word baseline (83.17%) | duration (s) | status |
|---|---|---|---|---|---|
| **relative** | 0.5251 | **86.7%** | **+3.5** | 230.2 | ok |
| rotary | 0.5345 | 86.4% | +3.2 | 230.7 | ok |
| alibi | 0.5853 | 84.6% | +1.4 | 218.8 | ok |
| t5_relative | 0.6154 | 83.7% | +0.5 | 234.1 | ok |
| learnable | 0.6053 | 83.7% | +0.5 | 247.0 | ok |
| **none (NoPE control)** | 0.6232 | 83.4% | **+0.2 (statistical noise)** | 270.8 | ok |
| absolute | 0.7457 | 78.7% | **-4.5** | 256.5 | ok |
| tape | 0.8292 | 75.8% | -7.4 | 242.9 | ok |
| learned | 1.1942 | 61.1% | -22.1 | 236.3 | ok |

## Reading this table (revised after adding the NoPE control)

The original framing of this experiment - "does PE choice affect POS-tagging accuracy" - undersold what's actually visible here. With the per-word baseline in hand, the eight schemes split into three tiers:

1. **Genuinely using position/context** (`relative`, `rotary`, `alibi`): clearly beat the trivial per-word lookup, meaning they're resolving at least some of the ~15-20% of tokens whose correct tag depends on surrounding words - the actual point of a sequence-aware POS tagger.
2. **Indistinguishable from memorizing word identity** (`t5_relative`, `learnable`, and **NoPE**): land within noise of the 83.17% baseline. NoPE landing here is the tell - a model with *zero* positional signal, by construction, cannot do anything but this, so any scheme scoring the same as NoPE is not demonstrably using position for this task at this budget.
3. **Worse than doing nothing** (`absolute`, `tape`, `learned`): these don't just fail to use position productively, their positional signal actively interferes with what a model could achieve from word identity alone - the same large/non-adaptive-perturbation problem diagnosed in the classification collapse (`classification-results-summary.md`), showing up here as a quality ceiling rather than a total collapse, because POS tagging's dense per-token supervision is a much stronger training signal than classification's one-label-per-sentence.

None of this was visible from the original 8-row table. See `nope-baseline-finding.md` for the full write-up of why the NoPE control was added and what it changes across all three tasks.
