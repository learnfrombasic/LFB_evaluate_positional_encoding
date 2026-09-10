# Lesson: the positional-encoding bug that runs without error

This is a standalone teaching write-up of a real bug found and fixed in this
codebase this session. It's presented as a lesson because it is, in my
experience, the single most common mistake in from-scratch positional
encoding implementations — and it is dangerous specifically *because*
nothing about it looks wrong. The model builds. It trains. The loss goes
down. You could ship it.

## The setup

BERT's original design adds one thing to the token embedding before the
first transformer layer: a positional signal, so that "the token *cat* at
position 3" is distinguishable from "the token *cat* at position 30." That
pattern — compute a positional table, add it to the embedding, once — is
exactly right for the scheme BERT actually used (fixed sinusoidal, or a
learned per-position table).

RoPE (Su et al., 2021) and Shaw et al.'s (2018) relative attention are
*not* that pattern, even though the temptation to implement them that way is
strong: you already have a "where does positional information go" slot in
your embeddings module, and if you're not looking closely at each paper's
actual equation, it's the obvious place to add one more positional scheme.

## The mistake, concretely

Before this session's fix, `RotaryPositionalEncoding` and
`RelativePositionalEncoding` were both instantiated inside `BertEmbeddings`
and applied once, to `word_embedding + token_type_embedding`, exactly like
the sinusoidal scheme next to them:

```python
# embeddings.py, before the fix
x = word_emb + tok_type_emb
x = self.position_embeddings(x)   # RoPE or Relative, called here
```

This compiles. It runs. Loss decreases during training, because *some*
positional signal is present and the model can exploit it. Nothing in a
training log tells you this is wrong.

## Why it's wrong: RoPE

RoPE's entire reason for existing is one property: after rotating a query
vector at position `i` and a key vector at position `j`, their dot product
`q_i · k_j` depends **only on `i - j`**, never on `i` and `j` individually.
That's what "relative" means in "relative position": shift a sentence by 10
tokens and the attention pattern between any two tokens is unchanged.

That property is a fact about the rotation applied to `q` and `k` — the
actual vectors that go into the attention dot product, inside each layer,
after the `W_Q`/`W_K` projections. If you instead rotate the *embedding*,
once, before any projection:

1. `W_Q` and `W_K` are two different, independently-learned linear maps.
   Rotating the shared input `x` and then applying two different projections
   does not preserve the rotation relationship between the resulting `q`
   and `k` — the projections can (and, once trained, will) mix rotated
   dimensions in ways that undo the geometry the rotation was supposed to
   create.
2. Layer 2 and beyond receive `x` that has already been transformed by
   attention and a feed-forward block. Whatever rotation structure survived
   layer 1 is gone by construction — nothing re-applies it.

The result: a real positional signal is present (so training isn't broken),
but the specific *relative-invariance* property RoPE is named for — the
reason to prefer it over a plain sinusoidal table — never actually exists in
the trained model.

**How to see this instead of reason about it**: `notebooks/pe_visualizations.ipynb`
§3 plots `q_i · k_j` for a fixed relative offset across many absolute
positions, for the *fixed* implementation. It's a flat line — the defining
property, made visible. If you ran that same check against the old
embedding-level version, it would not be flat.

## Why it's wrong: Shaw et al.'s relative attention

The mistake here was more severe: the original code computed a full
pairwise `(seq_len, seq_len, head_dim)` table of relative-position
embeddings — one vector per `(query position, key position)` pair — and then
**averaged it down to one vector per query position** before adding it to
the embedding:

```python
# relative.py, before the fix
pos_encoding = rel_pos_emb.mean(dim=1)   # (T, T, D) -> (T, D)
x = x + pos_encoding
```

Shaw et al.'s method is specifically about injecting a *pairwise* bias —
"how token `i` should attend to token `j` depends on `i - j`" — directly
into the attention score. Averaging across all keys before it ever reaches
attention collapses exactly the structure the method is named for. What
remained was, functionally, a third flavor of learned per-position
embedding, wearing the name "relative" without the mechanism.

The fix (`src/models/layers/positional_encoding/relative.py`) keeps the
pairwise table and adds it directly into the attention score computation,
per query-key pair, inside `ScaledDotProductAttention` — never collapsed,
never touching the embedding.

## The general lesson

Before writing code for a paper's positional-encoding scheme, answer one
question first: **does this paper's equation modify the token embedding, or
does it modify the attention score?** Every scheme in this codebase answers
that question differently:

| Scheme | Modifies | Because |
|---|---|---|
| Absolute / sinusoidal, learnable | the embedding, once | position is a per-token attribute, same at every layer |
| RoPE | q/k, inside every attention layer | the relative-invariance property is a fact about the attention dot product, not the embedding |
| Relative (Shaw et al.), ALiBi, T5-bucketed | the attention score, inside every attention layer | these are explicitly pairwise (query, key) biases, not per-token attributes |

Both integration points are one line of code to wire up. Only one of them is
what the paper means. The fastest way to tell which one you need: find the
equation in the paper and check which symbol it's added to — the input
embedding `x`, or the attention logit `q_i · k_j^T`.

## What the fix actually changed, empirically

This isn't just a theoretical concern — the four studies run this session
after the fix all show the corrected schemes behaving the way the papers
predict, on the very first runs after the rewrite:

- **MLM** (`results-summary.md`): the four attention-level schemes
  (including the fixed RoPE and relative) beat every embedding-level scheme,
  ~46% lower perplexity on average.
- **Length generalization** (`length-generalization.md`): RoPE and relative
  run correctly at 4x their training length with only mild degradation.
  This is the property that would be false if the bug were still present —
  an embedding-level "RoPE" has no mechanism to generalize to unseen
  positions any better than plain absolute encoding.
- **POS tagging** (`pos-tagging-results-summary.md`): relative and rotary
  are the top two schemes (86.7%, 86.4%).

None of that is proof by itself (small models, short runs — see each
report's honesty section), but it's consistent, in every task, with the
mechanism actually being present now, where it wasn't before.
