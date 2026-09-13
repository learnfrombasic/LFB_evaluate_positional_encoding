# PE comparative experiment summary: CoNLL-2003 NER

Same 11 PE schemes as the other task comparisons, trained from scratch on a 3000-sentence subset of CoNLL-2003 (same sentences as the POS-tagging runner, `ner_tags` instead of `pos_tags`), evaluated on the full 3250-sentence validation split. 9 BIO classes - a majority-class ('O') baseline should be computed empirically before reading these numbers as evidence of anything, the same caveat nope-baseline-finding.md raises for POS tagging.

| PE type | loss | accuracy | duration (s) | status |
|---|---|---|---|---|
| absolute | 0.4749 | 87.5% | 207.9 | ok |
| tape | 0.4630 | 87.6% | 239.3 | ok |
| learnable | 0.3614 | 91.3% | 243.2 | ok |
| learned | 0.5302 | 86.0% | 237.3 | ok |
| rotary | 0.2879 | 93.4% | 245.1 | ok |
| relative | 0.2959 | 93.3% | 202.7 | ok |
| alibi | 0.3877 | 91.4% | 177.8 | ok |
| t5_relative | 0.3895 | 90.7% | 175.8 | ok |
| kerple | 0.3880 | 91.3% | 173.9 | ok |
| xpos | 0.2881 | 93.4% | 189.7 | ok |
| none | 0.4016 | 90.6% | 200.1 | ok |
