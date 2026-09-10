# Length generalization: train short (64 tokens), test long (64 / 128 / 256)

Reuses the checkpoints from the main PE comparison (no retraining) - see `results-summary.md`. Each scheme was trained at 64 tokens; evaluated here on packed multi-sentence passages built from the local test shard at 64 (in-distribution), 128 (2x), and 256 (4x) tokens.

| Scheme | Layer | 64 tok (trained) | 128 tok (2x) | 256 tok (4x) |
|---|---|---|---|---|
| absolute | embedding | ppl 13937 / acc 6.0% | **FAILS** (RuntimeError: The size of tensor a (128) must match the size of tensor) | **FAILS** (RuntimeError: The size of tensor a (256) must match the size of tensor) |
| tape | embedding | ppl 14789 / acc 4.2% | **FAILS** (RuntimeError: The size of tensor a (128) must match the size of tensor) | **FAILS** (RuntimeError: The size of tensor a (256) must match the size of tensor) |
| learnable | embedding | ppl 10839 / acc 25.0% | **FAILS** (RuntimeError: The size of tensor a (128) must match the size of tensor) | **FAILS** (RuntimeError: The size of tensor a (256) must match the size of tensor) |
| learned | embedding | ppl 27488 / acc 0.4% | **FAILS** (RuntimeError: The size of tensor a (128) must match the size of tensor) | **FAILS** (RuntimeError: The size of tensor a (256) must match the size of tensor) |
| rotary | attention | ppl 8441 / acc 26.5% | ppl 9057 / acc 25.4% | ppl 9202 / acc 24.7% |
| relative | attention | ppl 8335 / acc 30.3% | ppl 8337 / acc 30.5% | ppl 8838 / acc 29.3% |
| alibi | attention | ppl 8825 / acc 25.3% | ppl 9132 / acc 25.1% | ppl 9254 / acc 24.9% |
| t5_relative | attention | ppl 8816 / acc 24.0% | ppl 9030 / acc 23.8% | ppl 9041 / acc 24.1% |
| **none (NoPE control)** | *(none)* | ppl 8527 / acc 26.3% | ppl 8766 / acc 25.9% | ppl 9161 / acc 25.2% |

## Reading this table

The embedding-level schemes hold a position-embedding table sized exactly to the 64-token training length - positions beyond 64 don't exist in that table, so a forward pass at 128 or 256 tokens is a shape error, not a worse score. The attention-level schemes compute their positional signal from the actual sequence length at every call, so they keep running - quality still degrades since they only ever trained on 64-token attention patterns, but they don't fall over. This is the concrete, task-relevant reason RoPE and ALiBi are the standard choice for anything that must handle variable or growing context length (chat, long documents, streaming) while absolute/learned position embeddings hard-cap a model's usable length at whatever it was trained with.

**NoPE added after the fact** (see `nope-baseline-finding.md`): it "generalizes" to any length trivially and for the least interesting possible reason - an identity function has no length-dependent parameters to run out of. Its degradation curve (8527 -> 8766 -> 9161 ppl) is nearly indistinguishable from `rotary`'s and `alibi`'s, reinforcing the same point the MLM summary makes: this table shows which schemes *structurally support* arbitrary length, not which ones are using the extra context productively. `relative` is the one scheme here that both survives length growth *and* clearly beats the NoPE floor in the main comparison (`results-summary.md`) - the closest this project gets to "uses long context and doesn't fall over."
