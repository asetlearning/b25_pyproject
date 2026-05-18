# Created by: Alex Myasnikov
# Date: May 15, 2026

"""Shared helpers for PatternBoost experiment scripts and the workflow loop."""

import json
from pathlib import Path

import tcgraph_ext as _ext

from algorithms.word_conversion import word_to_string
from algorithms.word_problem import word_from_str
from algorithms.pattern_boost.pb_tcgraph import (
    EditScorer,
    HammingScorer,
    HellingerScorer,
)


def make_scorer(score_type: str, challenges, num_generators: int):
    """Build a WordScorer by name.

    score_type: 'hellinger' | 'hamming' | 'edit' (case-insensitive).
    """
    t = score_type.lower()
    if t == 'hellinger':
        return HellingerScorer(challenges, num_generators)
    if t == 'hamming':
        return HammingScorer(challenges)
    if t == 'edit':
        return EditScorer(challenges)
    raise ValueError(f'unknown scorer type: {score_type}')


def load_challenges(path) -> tuple[list, dict[str, str]]:
    """Load challenges from relators_io-style JSON.

    Accepts both string ("ABaB...") and int-array ([-1,-2,...]) "word" fields.
    Returns (challenge Words list, name -> word_str mapping). Callers needing
    only the Word list can take `load_challenges(path)[0]`.
    """
    data = json.loads(Path(path).read_text())
    words: list = []
    name_to_str: dict[str, str] = {}
    for name, entry in data.items():
        w = entry['word']
        if isinstance(w, str):
            words.append(word_from_str(w))
            name_to_str[name] = w
        else:
            arr = list(w)
            words.append(_ext.Word(arr))
            name_to_str[name] = word_to_string(arr)
    return words, name_to_str
