"""
Tests for word conversion utilities.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

import pytest
import random

from algorithms.word_conversion import (
    int_to_char,
    word_to_string,
    char_to_int,
    string_to_word,
    word_power,
    freely_reduce,
    cyclically_reduce,
)


def test_int_to_char_positive():
    """Test conversion of positive integers."""
    assert int_to_char(1) == 'a'
    assert int_to_char(2) == 'b'


def test_int_to_char_negative():
    """Test conversion of negative integers."""
    assert int_to_char(-1) == 'A'
    assert int_to_char(-2) == 'B'


def test_int_to_char_invalid():
    """Test invalid integers raise ValueError."""
    with pytest.raises(ValueError):
        int_to_char(0)
    with pytest.raises(ValueError):
        int_to_char(3)
    with pytest.raises(ValueError):
        int_to_char(-3)


def test_word_to_string():
    """Test conversion of word array to string."""
    word = [-1, -2, 1, -2, -1, 2, 1]
    assert word_to_string(word) == "ABaBAba"

    # Additional test cases
    assert word_to_string([1, 2]) == "ab"
    assert word_to_string([-1, -2]) == "AB"
    assert word_to_string([]) == ""


def test_char_to_int():
    """Test conversion of characters to integers."""
    assert char_to_int('a') == 1
    assert char_to_int('b') == 2
    assert char_to_int('A') == -1
    assert char_to_int('B') == -2


def test_char_to_int_invalid():
    """Test invalid characters raise ValueError."""
    with pytest.raises(ValueError):
        char_to_int('c')
    with pytest.raises(ValueError):
        char_to_int('1')
    with pytest.raises(ValueError):
        char_to_int('')


def test_string_to_word():
    """Test conversion of string to word array."""
    assert string_to_word("ABaBAbaa") == [-1, -2, 1, -2, -1, 2, 1, 1]
    assert string_to_word("ab") == [1, 2]
    assert string_to_word("AB") == [-1, -2]
    assert string_to_word("") == []


def test_roundtrip():
    """Test roundtrip conversions with random words."""
    random.seed(42)

    for _ in range(20):
        # Generate random word
        length = random.randint(1, 50)
        word = [random.choice([-2, -1, 1, 2]) for _ in range(length)]

        # Convert to string and back
        s = word_to_string(word)
        back = string_to_word(s)

        assert word == back, f"Roundtrip failed for {word}"


def test_inverse_functions():
    """Test that char_to_int and int_to_char are inverses."""
    for n in [-2, -1, 1, 2]:
        assert char_to_int(int_to_char(n)) == n

    for c in ['a', 'b', 'A', 'B']:
        assert int_to_char(char_to_int(c)) == c


def test_word_power_integer():
    assert word_power("ab", 3) == "ababab"
    assert word_power("ab", 1) == "ab"
    assert word_power("ab", 0) == ""


def test_word_power_fractional():
    assert word_power("abab", 3.5) == "abababababababab"[:14]
    assert word_power("ab", 2.5) == "ababa"


def test_word_power_non_integer_length_raises():
    with pytest.raises(ValueError, match="must be an integer"):
        word_power("abc", 1.5)  # 1.5 * 3 = 4.5 — not integer


def test_word_power_negative_raises():
    with pytest.raises(ValueError, match=">= 0"):
        word_power("ab", -1)


def test_word_power_length():
    for n in [2, 3, 4]:
        assert len(word_power("ab", n)) == 2 * n
    assert len(word_power("abab", 3.5)) == 14


def test_freely_reduce_empty():
    assert freely_reduce("") == ""


def test_freely_reduce_no_cancellation():
    assert freely_reduce("abab") == "abab"
    assert freely_reduce("AB") == "AB"  # A,B are not inverses of each other
    assert freely_reduce("abAb") == "abAb"


def test_freely_reduce_single_pair():
    assert freely_reduce("aA") == ""
    assert freely_reduce("Aa") == ""
    assert freely_reduce("bB") == ""
    assert freely_reduce("Bb") == ""


def test_freely_reduce_nested():
    assert freely_reduce("aAbB") == ""   # aA→"", bB→""
    assert freely_reduce("abBA") == ""   # bB cancels → "aA" → ""
    assert freely_reduce("abaAb") == "abb"  # aA at pos 2-3 cancels


def test_freely_reduce_middle():
    # aAbAb: aA cancels → bAb (A and b not inverses)
    assert freely_reduce("aAbAb") == "bAb"


def test_cyclically_reduce_already_reduced():
    assert cyclically_reduce("abab") == "abab"
    assert cyclically_reduce("abaB") == "abaB"  # first=a, last=B: not inverses


def test_cyclically_reduce_free_reduces_first():
    assert cyclically_reduce("aAbB") == ""


def test_cyclically_reduce_cyclic_cancellation():
    # "abA": freely reduced; first=a, last=A → inverses → cancel → "b"
    assert cyclically_reduce("abA") == "b"
    # "Abba": freely reduced; first=A, last=a → inverses → cancel → "bb" → done
    assert cyclically_reduce("Abba") == "bb"


def test_cyclically_reduce_no_cyclic_cancel_after_free():
    # "abBa": bB cancels → "aa"; first=a, last=a: not inverses → "aa"
    assert cyclically_reduce("abBa") == "aa"


def test_cyclically_reduce_multilevel():
    # abbAbBABBA:
    #   freely reduce → abbAABBA
    #   a/A cancel → bbAABB → b/B cancel → bAAB → b/B cancel → AA
    assert cyclically_reduce("abbAbBABBA") == "AA"


def test_cyclically_reduce_empty():
    assert cyclically_reduce("") == ""


def test_cyclically_reduce_single():
    assert cyclically_reduce("a") == "a"


def test_word_power_fractional_fp_noise():
    # sub_len=5098, match_len=2550 → p*5098 = 22942 but float gives 22942.000000000004
    sub_len = 5098
    match_len = 2550
    p = 4 + match_len / sub_len
    word = "a" * sub_len
    result = word_power(word, p)
    assert len(result) == 22942
