# PE comparative experiment summary: GLUE RTE (NLI)

Same 11 PE schemes as the other task comparisons, fine-tuned from scratch (no MLM pretraining reused) on the full 2490-row RTE train split, evaluated on the full 277-row validation split (GLUE's real test split has no public labels). Random-guess baseline is 50% (2 classes, roughly balanced).

| PE type | loss | accuracy | duration (s) | status |
|---|---|---|---|---|
| absolute | 0.7036 | 47.7% | 306.6 | ok |
| tape | 0.7006 | 52.0% | 330.7 | ok |
| learnable | 2.2207 | 54.2% | 447.3 | ok |
| learned | 0.9211 | 47.3% | 531.8 | ok |
| rotary | 2.3185 | 50.5% | 487.8 | ok |
| relative | 2.4634 | 53.4% | 520.3 | ok |
| alibi | 2.3005 | 51.6% | 496.5 | ok |
| t5_relative | 2.3003 | 49.1% | 464.2 | ok |
| kerple | 2.3387 | 50.2% | 476.1 | ok |
| xpos | 2.3237 | 52.7% | 511.0 | ok |
| none | 2.1741 | 52.3% | 557.9 | ok |
