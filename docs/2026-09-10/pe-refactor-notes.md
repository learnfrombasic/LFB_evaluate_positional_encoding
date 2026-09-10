# Positional encoding: correctness fix + package refactor

## What was wrong

`src/models/layers/positional_encoding.py` was a single file implementing
every scheme, all wired into `BertEmbeddings` as an additive term on the
token embeddings (`x = position_embeddings(word_emb + type_emb)`). That is
correct for absolute/sinusoidal and learned/learnable encodings, but wrong
for two of the schemes this project exists to compare:

- **RoPE** (`RotaryPositionalEncoding`) was applied once, at the embedding
  layer, to the full hidden-size vector. RoPE's defining property — that the
  dot product of a rotated query and key depends only on their relative
  offset — only holds if the rotation happens on the actual Q/K vectors
  *inside* attention, per head, per layer. Rotating the embedding once
  before the first linear projection destroys that property immediately
  (`W_Q` and `W_K` mix the rotated dimensions arbitrarily) and never
  reintroduces position information in layer 2+. What was implemented was,
  functionally, just another additive sinusoidal-ish signal — not RoPE.
- **Relative** (`RelativePositionalEncoding`, Shaw et al. 2018) computed a
  full pairwise `(T, T, D)` relative-embedding tensor and then **averaged it
  down to one `(T, D)` vector per query position**, added once at the
  embedding layer. Shaw et al.'s method is specifically about injecting a
  *pairwise* (query, key) bias into the attention score itself; collapsing
  the pairwise table to a per-position average throws away exactly the
  structure that makes it "relative."
- ALiBi was already applied correctly (as a bias on attention scores), but
  recomputed its slope tensor from scratch on every forward call of every
  layer — wasted work, and a copy-paste of the slope formula that diverged
  from the reference implementation for non-power-of-2 head counts.
- `_get_relative_position_bucket` / `get_relative_positions` (a T5-style
  bucketed relative-position bias, Raffel et al. 2019) were fully
  implemented but never wired into `get_pos_encoder` — dead code.

## References checked this session

- RoFormer / RoPE: https://arxiv.org/abs/2104.09864 — rotate-half
  formulation confirmed: `q̃ = q·cos(θ) + rotate_half(q)·sin(θ)`, applied to
  per-head query/key vectors.
- Shaw et al., Self-Attention with Relative Position Representations:
  https://aclanthology.org/N18-2074/ — bias added to the K-side attention
  score term via a clipped relative-position embedding table.
- ALiBi, Press et al.: https://arxiv.org/abs/2108.12409 — static per-head
  slope × negative distance added pre-softmax; slopes form a geometric
  sequence, computed once.
- T5 relative bias: https://arxiv.org/abs/1910.10683 (bucket function
  adapted from HuggingFace's T5 implementation, already present in this
  repo's dead code).

## What changed

`src/models/layers/positional_encoding.py` → package
`src/models/layers/positional_encoding/`, one file per scheme:

| File | Type | Level |
|---|---|---|
| `sinusoidal.py` | absolute / fixed / temporal / tape | embedding |
| `learnable.py` | learnable / learned | embedding |
| `rotary.py` | rotary / rope | **attention** (rewritten) |
| `relative.py` | relative | **attention** (rewritten) |
| `alibi.py` | alibi | attention (slopes precomputed + cached) |
| `t5_relative.py` | t5_relative (new, bonus) | attention (wired up from dead code) |
| `misc.py` | SineSPE / ConvSPE / VariablePositionalEncoding | unused by BERT, kept for reuse |

`get_pos_encoder(name)` now returns `(level, cls)`. `BertEmbeddings` only
instantiates a positional module when `level == "embedding"`.
`ScaledDotProductAttention` (in `attentions.py`) now owns attention-level
encoders: it builds one from config when `level == "attention"`, calls
`rotate_qk(q, k)` before the Q·K product for rotary, and adds
`attention_bias(seq_len, device, dtype, q)` into the attention scores after
scaling (before masking) for relative/alibi/t5_relative. No new base class
was introduced — the two behaviors are duck-typed (`hasattr(..., "rotate_qk")`
/ `hasattr(..., "attention_bias")`), matching the project's existing
no-premature-abstraction style.

The relative-K-side term only is implemented (no `a^V` value-side term from
the original paper) — this matches the widely used simplified variant
(e.g. HuggingFace BERT's `relative_key` position type) and keeps the
attention layer's return shape unchanged.

## Verification

`tests/models/layers/positional_encoding/test_*.py` — one file per scheme,
each asserting the property that actually matters for that scheme (RoPE:
dot product depends only on relative offset, verified through the public
`rotate_qk` API; ALiBi: slope count/geometric ratio + monotonic bias decay;
relative: clipped/symmetric bucket indices). `tests/models/test_attention_shapes.py`
runs a full `BertMLM` forward+backward for all 12 supported
`position_embedding_type` values (everything documented in `config.yaml`'s
comment, plus `t5_relative`) to catch registration/shape regressions.
All 29 tests pass on CPU (`pytest tests/ -q`), no GPU or network required.
