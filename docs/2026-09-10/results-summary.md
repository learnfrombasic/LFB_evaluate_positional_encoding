# PE comparative experiment summary

Short feasibility-scale comparison, run on CPU with a small model (hidden=256, 2 layers), a 2000-row train subset, and training.max_steps capped - see machine-eval.md and experiment-design.md for why. This ranks PE schemes directionally under this tiny budget; it is not a publication-grade pretraining result and small gaps between schemes are not conclusive.

| PE type | loss | perplexity | accuracy | duration (s) | status |
|---|---|---|---|---|---|
| absolute | 9.5081 | 13467.71 | 0.0386 | 129.1 | ok |
| tape | 9.5695 | 14321.51 | 0.0278 | 144.1 | ok |
| learnable | 9.2765 | 10684.50 | 0.2449 | 148.5 | ok |
| learned | 10.1017 | 24384.01 | 0.0139 | 127.5 | ok |
| rotary | 9.0565 | 8574.40 | 0.2576 | 121.2 | ok |
| relative | 8.9618 | 7799.22 | 0.3203 | 99.0 | ok |
| alibi | 9.0545 | 8557.32 | 0.2603 | 89.3 | ok |
| t5_relative | 9.0810 | 8787.02 | 0.2199 | 85.8 | ok |
| **none (NoPE control)** | 9.0565 | 8573.81 | 0.2576 | 67.3 | ok |

**NoPE added after the fact as a scientific control - see `nope-baseline-finding.md` for the full analysis.** It lands *statistically tied with `rotary`* (loss/perplexity/accuracy all match to 3+ significant figures) and beats `absolute`/`tape`/`learned` outright. For masked-token prediction specifically, a lot of the signal is apparently recoverable from bag-of-context alone - see the classification and POS-tagging summaries for tasks where this stops being true.
