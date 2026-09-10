# PE comparative experiment summary: SST-2 sentiment classification

Same 8 PE schemes as the MLM comparison, now fine-tuned from scratch (no MLM pretraining reused) on a 3000-row subset of GLUE SST-2, evaluated on the full 872-row validation split (GLUE's real test split has no public labels). Random-guess baseline is 50%.

| PE type | loss | accuracy | duration (s) | status |
|---|---|---|---|---|
| absolute | 0.6958 | 50.9% | 173.9 | ok |
| tape | 0.6954 | 52.9% | 164.8 | ok |
| learnable | 1.0775 | 72.5% | 146.7 | ok |
| learned | 0.6970 | 50.9% | 134.9 | ok |
| rotary | 1.0684 | 72.6% | 139.0 | ok |
| relative | 1.1302 | 71.8% | 142.2 | ok |
| alibi | 1.0850 | 73.9% | 139.7 | ok |
| t5_relative | 1.1719 | 71.6% | 140.0 | ok |
| **none (NoPE control)** | 1.0808 | 72.5% | 209.3 | ok |

**NoPE added after the fact as a scientific control - see `nope-baseline-finding.md`.** It lands inside the "escaped the collapse" cluster (72.5%, tied with `learnable`), not with `absolute`/`tape`/`learned`. That means the earlier collapse was never really about *missing* positional information - it was specifically about a large, non-adaptive perturbation destabilizing training from a random init. Sentiment classification is well known to be substantially solvable from bag-of-words lexical cues alone, which is consistent with a model using *no* position at all reaching the same ceiling as the schemes that trained successfully.
