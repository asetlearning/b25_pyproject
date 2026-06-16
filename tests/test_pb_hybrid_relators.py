# Created by: Alex Myasnikov
# Date: June 9, 2026

import tcgraph_ext as _ext
from algorithms.pattern_boost.relators import build_hybrid_relators
from algorithms.pattern_boost.pb_tcgraph import build_standard_relators


def _letters(words):
    return [tuple(w.get_letters()) for w in words]


def test_none_standard_equals_build_standard_relators():
    for n in (2, 3):
        expected = _letters(build_standard_relators(2, n))
        got = _letters(build_hybrid_relators(2, n, None))
        assert sorted(expected) == sorted(got)


def test_all_standard_equals_build_standard_relators():
    n = 3
    expected = _letters(build_standard_relators(2, n))
    got = _letters(build_hybrid_relators(2, n, list(range(1, n + 1))))
    assert sorted(expected) == sorted(got)


def test_nonstandard_generators_add_5th_powers():
    # generators 1,2 standard; 3,4,5 non-standard
    result = _letters(build_hybrid_relators(2, 5, [1, 2]))
    for g in (3, 4, 5):
        assert tuple([g] * 5) in result
        assert tuple([-g] * 5) in result


def test_no_duplicates():
    result = _letters(build_hybrid_relators(2, 5, [1, 2]))
    assert len(result) == len(set(result))


def test_remapping_correct():
    # standard generators are [2, 4] (not the generic {1,2})
    result = _letters(build_hybrid_relators(1, 4, [2, 4]))
    # standard part: 5th powers of single generators 2 and 4
    assert tuple([2] * 5) in result
    assert tuple([4] * 5) in result
    assert tuple([-2] * 5) in result
    assert tuple([-4] * 5) in result
    # non-standard: generators 1 and 3
    assert tuple([1] * 5) in result
    assert tuple([-1] * 5) in result
    assert tuple([3] * 5) in result
    assert tuple([-3] * 5) in result
    # standard relators should only use letters {2,-2,4,-4} — no generic {1,2} leaking through
    standard_relators_raw = _letters(build_standard_relators(1, 2))
    valid_letters = {2, -2, 4, -4}
    standard_in_result = [t for t in result if all(l in valid_letters for l in t)]
    expected_standard = sorted([
        tuple(2 if l == 1 else (-2 if l == -1 else (4 if l == 2 else -4)) for l in t)
        for t in standard_relators_raw
    ])
    assert sorted(standard_in_result) == expected_standard


def test_single_standard_generator():
    # 1 standard generator out of 3 total
    result = _letters(build_hybrid_relators(1, 3, [2]))
    # standard: g=2, (g^{-1})=-2, their 5th powers
    assert tuple([2] * 5) in result
    assert tuple([-2] * 5) in result
    # non-standard: 1, 3
    for g in (1, 3):
        assert tuple([g] * 5) in result
        assert tuple([-g] * 5) in result
    assert len(result) == len(set(result))
