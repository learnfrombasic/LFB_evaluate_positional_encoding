# Span-cutoff augmentation: does each scheme tolerate a gap in the sequence?

Same CoNLL-2003 POS-tagging setup, but every sentence (train and validation) has one contiguous span (~20% of its length) deleted before training. Unlike full word-shuffling, surviving tokens keep their original relative order and neighbors - only a discontinuity (like a redacted phrase, or two originally non-adjacent sentences stitched together) is introduced.

| PE type | clean accuracy | cutoff accuracy | delta | duration (s) | status |
|---|---|---|---|---|---|
| relative | 86.7% | 85.5% | -1.2pp | 232.9 | ok |
| rotary | 86.4% | 85.3% | -1.1pp | 208.3 | ok |
| absolute | 78.7% | 77.4% | -1.3pp | 221.4 | ok |
| none | 83.4% | 82.6% | -0.8pp | 260.1 | ok |
