"""
Tests for maximal power detection algorithm.

Created by: Alex Myasnikov
Date: February 12, 2026
"""

import pytest
import random
from algorithms.maximal_power import find_maximal_powers


# Basic functionality tests

def test_user_example():
    """User example: S=aaBB, W=bbaaBBaaBBaaBAAaa -> 2.75 power at pos 2."""
    result = find_maximal_powers("aaBB", "bbaaBBaaBBaaBAAaa")
    assert len(result) == 1
    assert result[0] == ("aaBBaaBBaaB", 2.75, 2)


def test_exact_two_powers():
    """Exactly 2.0 repetitions."""
    result = find_maximal_powers("ab", "AababB")
    assert len(result) == 1
    assert result[0] == ("abab", 2.0, 1)


def test_fractional_2_25():
    """2.25 repetitions (2 full + 1/4)."""
    result = find_maximal_powers("abBA", "abBAabBAa")
    assert len(result) == 1
    assert result[0][0] == "abBAabBAa"
    assert result[0][1] == 2.25
    assert result[0][2] == 0


def test_fractional_2_5():
    """2.5 repetitions (2 full + 1/2)."""
    result = find_maximal_powers("ab", "ababa")
    assert len(result) == 1
    assert result[0] == ("ababa", 2.5, 0)


def test_fractional_2_75():
    """2.75 repetitions (2 full + 3/4)."""
    result = find_maximal_powers("abBA", "abBAabBAabB")
    assert len(result) == 1
    assert result[0][0] == "abBAabBAabB"
    assert result[0][1] == 2.75
    assert result[0][2] == 0


def test_large_power():
    """Power with 5 full repetitions."""
    result = find_maximal_powers("ab", "abababababab")
    assert len(result) == 1
    assert result[0] == ("abababababab", 6.0, 0)


# Edge cases

def test_empty_subword():
    """Empty subword returns empty list."""
    result = find_maximal_powers("", "abcd")
    assert result == []


def test_empty_word():
    """Empty word returns empty list."""
    result = find_maximal_powers("ab", "")
    assert result == []


def test_both_empty():
    """Both empty returns empty list."""
    result = find_maximal_powers("", "")
    assert result == []


def test_subword_longer_than_word():
    """Subword longer than word returns empty list."""
    result = find_maximal_powers("abcd", "ab")
    assert result == []


def test_subword_not_in_word():
    """Subword not present in word returns empty list."""
    result = find_maximal_powers("AB", "ababaabb")
    assert result == []


def test_single_occurrence():
    """Single occurrence (power=1) is excluded."""
    result = find_maximal_powers("ab", "AabB")
    assert result == []


def test_subword_equals_word():
    """Subword equals word (power=1) is excluded."""
    result = find_maximal_powers("abBA", "abBA")
    assert result == []


# Multiple powers

def test_two_non_overlapping_powers():
    """Two separate maximal powers in one word."""
    result = find_maximal_powers("ab", "ababAabab")
    assert len(result) == 2
    assert ("abab", 2.0, 0) in result
    assert ("abab", 2.0, 5) in result


def test_three_powers_different_lengths():
    """Three powers with different lengths."""
    result = find_maximal_powers("a", "aabaaabaaaa")
    assert len(result) == 3
    # aa at pos 0 (power=2)
    assert ("aa", 2.0, 0) in result
    # aaa at pos 3 (power=3)
    assert ("aaa", 3.0, 3) in result
    # aaaa at pos 7 (power=4)
    assert ("aaaa", 4.0, 7) in result


def test_no_overlapping_results():
    """Ensure results don't overlap (covered_until tracking)."""
    result = find_maximal_powers("ab", "ababababab")
    # Should find one long power, not multiple overlapping ones
    assert len(result) == 1
    assert result[0] == ("ababababab", 5.0, 0)


# Boundary conditions

def test_power_at_start():
    """Power at position 0."""
    result = find_maximal_powers("ab", "ababABa")
    assert len(result) == 1
    assert result[0][2] == 0


def test_power_at_end():
    """Power at end of word."""
    result = find_maximal_powers("ab", "ABaabab")
    assert len(result) == 1
    assert result[0] == ("abab", 2.0, 3)


def test_fractional_at_word_end():
    """Fractional part ends exactly at word boundary."""
    result = find_maximal_powers("abB", "abBabBa")
    assert len(result) == 1
    substring, power, pos = result[0]
    assert substring == "abBabBa"
    assert abs(power - 2.333333) < 0.001  # 2 + 1/3
    assert pos == 0


def test_word_is_all_powers():
    """Entire word is a single maximal power."""
    result = find_maximal_powers("aB", "aBaBaBaB")
    assert len(result) == 1
    assert result[0] == ("aBaBaBaB", 4.0, 0)


# Character validation

def test_mixed_case_valid():
    """Mixed case a/A and b/B are valid."""
    result = find_maximal_powers("aB", "aaBaaBaaa")
    # "aaB" is not "aB", so no match
    result2 = find_maximal_powers("aB", "aBaBaBaB")
    assert len(result2) == 1
    assert result2[0] == ("aBaBaBaB", 4.0, 0)


def test_invalid_char_in_subword():
    """Invalid character in subword raises ValueError."""
    with pytest.raises(ValueError, match="Invalid character in subword"):
        find_maximal_powers("axb", "abBA")


def test_invalid_char_in_word():
    """Invalid character in word raises ValueError."""
    with pytest.raises(ValueError, match="Invalid character in word"):
        find_maximal_powers("ab", "abxBA")


def test_strict_alphabet():
    """Only {a,b,A,B} allowed."""
    valid_chars = ['a', 'b', 'A', 'B']
    invalid_chars = ['c', 'x', '1', ' ', '-']

    for char in valid_chars:
        # Should not raise
        result = find_maximal_powers(char, char + char)
        assert len(result) == 1  # power of 2

    for char in invalid_chars:
        with pytest.raises(ValueError):
            find_maximal_powers(char, char + char)


# Random tests

def test_random_known_powers():
    """Generate words with known powers and verify detection."""
    random.seed(42)  # Reproducibility

    def random_pattern(length):
        """Generate random pattern from {a,b,A,B}."""
        chars = ['a', 'b', 'A', 'B']
        return ''.join(random.choice(chars) for _ in range(length))

    for iteration in range(10):
        # Generate random subword
        sub_len = random.randint(2, 5)
        subword = random_pattern(sub_len)

        # Create known power (2-4 repetitions)
        reps = random.randint(2, 4)
        power_str = subword * reps

        # Add fractional part randomly
        frac_len = random.randint(0, sub_len - 1)
        if frac_len > 0:
            power_str += subword[:frac_len]

        # Add random padding
        prefix = random_pattern(random.randint(1, 3))
        suffix = random_pattern(random.randint(1, 3))
        word = prefix + power_str + suffix

        # Verify detection
        result = find_maximal_powers(subword, word)

        # Should find at least one power
        assert len(result) >= 1, f"Iteration {iteration}: No power found in '{word}' for subword '{subword}'"

        # Verify the known power is detected
        found = False
        for substring, power_count, start_pos in result:
            if substring == power_str and start_pos == len(prefix):
                found = True
                expected_power = reps + (frac_len / sub_len)
                assert abs(power_count - expected_power) < 0.001
                break

        assert found, f"Iteration {iteration}: Expected power '{power_str}' not found"


def test_overlapping_subword_occurrences():
    """Subword overlaps with itself - ensure correct handling."""
    # Pattern "aa" in "aaaa" should give one maximal power of 4.0
    result = find_maximal_powers("aa", "aaaa")
    assert len(result) == 1
    assert result[0] == ("aaaa", 2.0, 0)

    # Pattern "aaa" in "aaaaaa" should give one power of 2.0
    result = find_maximal_powers("aaa", "aaaaaa")
    assert len(result) == 1
    assert result[0] == ("aaaaaa", 2.0, 0)


def test_fractional_cutoff_mid_subword():
    """Fractional part stops mid-subword due to mismatch."""
    result = find_maximal_powers("abB", "abBabBaBaaa")
    assert len(result) == 1
    substring, power, pos = result[0]
    assert substring == "abBabBa"
    # 2 full + 1/3 (matches only "a" from "abB")
    assert abs(power - 2.333333) < 0.001
    assert pos == 0
