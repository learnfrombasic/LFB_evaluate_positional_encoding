"""Positional encoding registry for Transformer models.

Each scheme lives in its own module. Schemes are either:
  - "embedding"-level: added once to token embeddings before the encoder
    (absolute/sinusoidal, learnable/learned).
  - "attention"-level: applied inside every attention layer, operating on
    per-head query/key vectors or attention scores (rotary, relative,
    alibi, t5_relative).

`get_pos_encoder` returns `(level, cls)` so callers know where to
instantiate and invoke the encoder.
"""

from functools import partial

from .alibi import ALiBiPositionalEncoding
from .identity import NoPositionalEncoding
from .learnable import LearnablePositionalEncoding
from .misc import ConvSPE, SineSPE, VariablePositionalEncoding
from .relative import RelativePositionalEncoding
from .rotary import RotaryPositionalEncoding
from .sinusoidal import SinusoidalPositionalEncoding
from .t5_relative import T5RelativePositionalEncoding, get_relative_positions

__all__ = [
    "SinusoidalPositionalEncoding",
    "LearnablePositionalEncoding",
    "RotaryPositionalEncoding",
    "RelativePositionalEncoding",
    "ALiBiPositionalEncoding",
    "T5RelativePositionalEncoding",
    "NoPositionalEncoding",
    "SineSPE",
    "ConvSPE",
    "VariablePositionalEncoding",
    "get_relative_positions",
    "get_pos_encoder",
]

_EMBEDDING_LEVEL = {
    "sinusoidal": SinusoidalPositionalEncoding,
    "fixed": SinusoidalPositionalEncoding,
    "absolute": SinusoidalPositionalEncoding,
    "temporal": partial(SinusoidalPositionalEncoding, return_encoding_only=True),
    "tape": partial(SinusoidalPositionalEncoding, tAPE_mode=True),
    "learnable": LearnablePositionalEncoding,
    "learned": partial(LearnablePositionalEncoding, init_std=1.0),
    "none": NoPositionalEncoding,
    "nope": NoPositionalEncoding,
}

_ATTENTION_LEVEL = {
    "rotary": RotaryPositionalEncoding,
    "rope": RotaryPositionalEncoding,
    "relative": RelativePositionalEncoding,
    "alibi": ALiBiPositionalEncoding,
    "t5_relative": T5RelativePositionalEncoding,
}


def get_pos_encoder(pos_encoding: str):
    """Looks up a positional encoding by config key.

    Returns:
        (level, cls): level is "embedding" or "attention"; cls is the class
        (or partial) to instantiate.
    """
    key = pos_encoding.lower()
    if key in _EMBEDDING_LEVEL:
        return "embedding", _EMBEDDING_LEVEL[key]
    if key in _ATTENTION_LEVEL:
        return "attention", _ATTENTION_LEVEL[key]

    valid = sorted(set(_EMBEDDING_LEVEL) | set(_ATTENTION_LEVEL))
    raise ValueError(
        f"Unknown positional encoding: '{pos_encoding}'. Valid options: {valid}"
    )
