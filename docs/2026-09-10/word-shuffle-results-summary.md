# Word-order shuffle augmentation: does each scheme actually use word order?

Same CoNLL-2003 POS-tagging setup as `pos-tagging-results-summary.md`, but every sentence (train and validation) has its words - and their tags, moved together - randomly permuted before training. Word identity -> tag information is fully preserved; only real word order/adjacency is destroyed.

| PE type | clean accuracy | shuffled accuracy | delta | duration (s) | status |
|---|---|---|---|---|---|
| relative | 86.7% | 83.9% | -2.8pp | 241.0 | ok |
| rotary | 86.4% | 83.9% | -2.5pp | 222.8 | ok |
| absolute | 78.7% | 78.5% | -0.2pp | 252.0 | ok |
| none | 83.4% | 83.4% | -0.0pp | 323.1 | ok |
