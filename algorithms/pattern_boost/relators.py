# Created by: Alex Myasnikov
# Date: June 9, 2026

from typing import Optional

import tcgraph_ext as _ext

_pb = _ext.pattern_boost


def build_hybrid_relators(
    max_relator_len: int,
    total_generators: int,
    standard_generators: Optional[list[int]] = None,
) -> list:
    """Build relators for a hybrid generator set.

    Standard generators get full W^5 enumeration up to max_relator_len.
    Non-standard generators get only g^5 and (g^{-1})^5.

    Args:
        max_relator_len: max word length for standard generator enumeration
        total_generators: total number of generators (1..total_generators)
        standard_generators: 1-based indices of standard generators; None = all standard
    """
    if standard_generators is None:
        return _pb.build_standard_relators(max_relator_len, total_generators)

    k = len(standard_generators)
    base_relators = _pb.build_standard_relators(max_relator_len, k)

    remap = {}
    for i, g in enumerate(standard_generators, start=1):
        remap[i] = g
        remap[-i] = -g

    result = []
    seen = set()

    for w in base_relators:
        remapped = _ext.Word([remap[l] for l in w.get_letters()])
        key = tuple(remapped.get_letters())
        if key not in seen:
            seen.add(key)
            result.append(remapped)

    standard_set = set(standard_generators)
    for g in range(1, total_generators + 1):
        if g not in standard_set:
            for letters in ([g] * 5, [-g] * 5):
                w = _ext.Word(letters)
                key = tuple(w.get_letters())
                if key not in seen:
                    seen.add(key)
                    result.append(w)

    return result
