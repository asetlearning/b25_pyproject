# Created by: Alex Myasnikov
# Date: May 15, 2026

"""Shared helpers for PatternBoost experiment scripts and the workflow loop."""

import json
import random
from pathlib import Path
from typing import Optional

import tcgraph_ext as _ext

from algorithms.word_conversion import word_to_string
from algorithms.word_problem import word_from_str
from algorithms.pattern_boost.pb_tcgraph import (
    AbelianizedProductLengthScorer,
    EditScorer,
    HammingScorer,
    HellingerScorer,
    HighPowerCountScorer,
    HybridScorer,
    LengthRangeScorer,
    WordProductLengthScorer,
    make_hybrid_scorer,
)


def make_scorer(score_type: str, challenges, num_generators: int, config=None, components=None, params=None):
    """Build a WordScorer by name.

    score_type: hellinger | hamming | edit | word_product_length |
                abelianized_product_length | high_power_count | hybrid (case-insensitive).
    config: outer ScorerConfig (compressor + reduce2_5); applied to the top-level scorer only.
    components: required when score_type=='hybrid'. List of dicts:
                {type: str, weight: float, params?: {q?: int, threshold?: float}}
                Sub-scorers always use default ScorerConfig — decoding is handled once by the
                outer HybridScorer; passing config to sub-scorers would double-decode words.
    """
    extra = (config,) if config is not None else ()
    t = score_type.lower()
    if t == 'hellinger':
        return HellingerScorer(challenges, num_generators, *extra)
    if t == 'hamming':
        return HammingScorer(challenges, *extra)
    if t == 'edit':
        return EditScorer(challenges, *extra)
    if t == 'word_product_length':
        return WordProductLengthScorer(challenges, *extra)
    if t == 'abelianized_product_length':
        return AbelianizedProductLengthScorer(challenges, num_generators, *extra)
    if t == 'high_power_count':
        p = params or {}
        threshold = float(p.get('threshold', 2.5))
        return HighPowerCountScorer(threshold, *extra)
    if t == 'length_range':
        p = params or {}
        return LengthRangeScorer(
            float(p.get('min_length', 10.0)),
            float(p.get('max_length', 100.0)),
            float(p.get('sigma', 10.0)),
            bool(p.get('reduce_before_scoring', False)),
            *extra)
    if t == 'hybrid':
        if not components:
            raise ValueError('hybrid scorer requires a non-empty components list')
        pairs = []
        for comp in components:
            sub_type = comp['type']
            weight = float(comp.get('weight', 1.0))
            params = comp.get('params') or {}
            sub = _make_sub_scorer(sub_type, challenges, num_generators, params)
            pairs.append((sub, weight))
        return make_hybrid_scorer(pairs, *extra)
    raise ValueError(f'unknown scorer type: {score_type!r}')


def _make_sub_scorer(score_type: str, challenges, num_generators: int, params: dict):
    """Build a sub-scorer for HybridScorer with default ScorerConfig."""
    t = score_type.lower()
    if t == 'hellinger':
        return HellingerScorer(challenges, num_generators)
    if t == 'hamming':
        return HammingScorer(challenges)
    if t == 'edit':
        return EditScorer(challenges)
    if t == 'word_product_length':
        return WordProductLengthScorer(challenges)
    if t == 'abelianized_product_length':
        q = int(params.get('q', 5))
        return AbelianizedProductLengthScorer(challenges, num_generators, q)
    if t == 'high_power_count':
        threshold = float(params.get('threshold', 2.5))
        return HighPowerCountScorer(threshold)
    if t == 'length_range':
        return LengthRangeScorer(
            float(params.get('min_length', 10.0)),
            float(params.get('max_length', 100.0)),
            float(params.get('sigma', 10.0)),
            bool(params.get('reduce_before_scoring', False)))
    raise ValueError(f'unknown sub-scorer type: {score_type!r}')


def load_challenges(
    path,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
    names_filter: Optional[list] = None,
) -> tuple[list, dict[str, str]]:
    """Load challenges from relators_io-style JSON.

    Accepts both string ("ABaB...") and int-array ([-1,-2,...]) "word" fields.
    Returns (challenge Words list, name -> word_str mapping). Callers needing
    only the Word list can take `load_challenges(path)[0]`.

    Priority: names_filter > sample_size. If `names_filter` is given, only
    challenges whose name appears in that list are loaded (order preserved).
    Otherwise, if `sample_size` is given and smaller than total, a seeded
    random subset is returned.
    """
    data = json.loads(Path(path).read_text())
    items = list(data.items())
    if names_filter is not None:
        allowed = set(names_filter)
        items = [(n, e) for n, e in items if n in allowed]
    elif sample_size is not None and sample_size < len(items):
        items = random.Random(seed).sample(items, sample_size)
    words: list = []
    name_to_str: dict[str, str] = {}
    for name, entry in items:
        w = entry['word']
        if isinstance(w, str):
            words.append(word_from_str(w))
            name_to_str[name] = w
        else:
            arr = list(w)
            words.append(_ext.Word(arr))
            name_to_str[name] = word_to_string(arr)
    return words, name_to_str
