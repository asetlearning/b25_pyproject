# Created by: Alex Myasnikov
# Date: March 19, 2026
import pytest

from algorithms.word_conversion import minimal_period, cyclically_permute
from algorithms.greedy_reduction import greedy_reduce, greedy_reduce_cyclic, greedy_reduce_cyclic_exhaustive


# ---------------------------------------------------------------------------
# minimal_period tests
# ---------------------------------------------------------------------------

def test_minimal_period_single_char():
    assert minimal_period("AAAA") == "A"

def test_minimal_period_two_char():
    assert minimal_period("aBaB") == "aB"

def test_minimal_period_no_period():
    assert minimal_period("abba") == "abba"

def test_minimal_period_itself():
    assert minimal_period("ab") == "ab"

def test_minimal_period_length_one():
    assert minimal_period("A") == "A"

def test_minimal_period_triple():
    assert minimal_period("abcabc") == "abc"


# ---------------------------------------------------------------------------
# greedy_reduce — fifth-power reduction
# ---------------------------------------------------------------------------

def test_reduce_AAAAA():
    assert greedy_reduce("AAAAA", "test", show_progress=False) == ""

def test_reduce_BBBBB():
    assert greedy_reduce("BBBBB", "test", show_progress=False) == ""

def test_reduce_aBaBaBaBaB():
    assert greedy_reduce("aBaBaBaBaB", "test", show_progress=False) == ""

def test_reduce_fifth_power_with_suffix():
    result = greedy_reduce("AAAAAb", "test", show_progress=False)
    assert result == "b"

def test_reduce_fifth_power_with_prefix():
    result = greedy_reduce("bAAAAA", "test", show_progress=False)
    assert result == "b"

def test_no_reduction_needed():
    word = "ab"
    result = greedy_reduce(word, "test", show_progress=False)
    assert result == word

def test_reduce_aaaaa():
    assert greedy_reduce("aaaaa", "test", show_progress=False) == ""


# ---------------------------------------------------------------------------
# cyclically_permute
# ---------------------------------------------------------------------------

def test_cyclically_permute_default():
    assert cyclically_permute("abcde") == "bcdea"

def test_cyclically_permute_by_two():
    assert cyclically_permute("abcde", 2) == "cdeab"

def test_cyclically_permute_zero():
    assert cyclically_permute("abcde", 0) == "abcde"

def test_cyclically_permute_full_rotation():
    assert cyclically_permute("abcde", 5) == "abcde"

def test_cyclically_permute_empty():
    assert cyclically_permute("") == ""


# ---------------------------------------------------------------------------
# greedy_reduce_cyclic
# ---------------------------------------------------------------------------

def test_greedy_reduce_cyclic_fifth_power():
    assert greedy_reduce_cyclic("AAAAA", "test", show_progress=False) == ""

def test_greedy_reduce_cyclic_finds_trivial_via_permutation():
    # AAAAA is trivial; any cyclic permutation is still AAAAA
    result = greedy_reduce_cyclic("AAAAA", "test", show_progress=False)
    assert result == ""

def test_greedy_reduce_cyclic_no_reduction():
    result = greedy_reduce_cyclic("ab", "test", show_progress=False)
    assert len(result) <= 2

def test_greedy_reduce_cyclic_returns_minimal():
    word = "bAAAAA"
    direct = greedy_reduce(word, "test", show_progress=False)
    cyclic = greedy_reduce_cyclic(word, "test", show_progress=False)
    assert len(cyclic) <= len(direct)


# ---------------------------------------------------------------------------
# greedy_reduce_cyclic_exhaustive
# ---------------------------------------------------------------------------

def test_greedy_reduce_cyclic_exhaustive_trivial():
    assert greedy_reduce_cyclic_exhaustive("AAAAA", "test", show_progress=False) == ""

def test_greedy_reduce_cyclic_exhaustive_no_reduction():
    result = greedy_reduce_cyclic_exhaustive("ab", "test", show_progress=False)
    assert len(result) <= 2

def test_greedy_reduce_cyclic_exhaustive_no_longer_than_cyclic():
    word = "bAAAAA"
    cyclic = greedy_reduce_cyclic(word, "test", show_progress=False)
    exhaustive = greedy_reduce_cyclic_exhaustive(word, "test", show_progress=False)
    assert len(exhaustive) <= len(cyclic)


# ---------------------------------------------------------------------------
# single-char base reductions (power 3, requires 2-char square detection)
# ---------------------------------------------------------------------------

def test_single_char_power_reduction():
    """Words with aaa/bbb patterns (power 3) should reduce."""
    words = [
        "AAbaaabAAbaaabAAb",
        "baabbAAABBBaabaaaBaaaBAABaaabaab",
        "baBBBabbaabbaBBBabbabbaBaBBBab",
    ]
    for w in words:
        result = greedy_reduce(w, name=w[:8], show_progress=False)
        assert len(result) < len(w), f"Expected reduction of '{w}', got '{result}'"


# ---------------------------------------------------------------------------
# power > 5 reductions
# ---------------------------------------------------------------------------

def test_reduce_aaaaaa():
    """power 6 of single char: aaaaaa -> a (W^5=identity leaves W^1)"""
    assert greedy_reduce("aaaaaa", "test", show_progress=False) == "a"

def test_reduce_power_gt5_real_example():
    """Real-life word containing aaaaaa; must be shortened."""
    word = "aabABABAABabAAAbaabaBAABABAbaaaaaabaaaa"
    result = greedy_reduce(word, "test", show_progress=False)
    assert len(result) < len(word)
