"""
Tests for word squares detection algorithm.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

import pytest
import random
from algorithms.word_squares import find_squares


def test_empty_string():
    """Empty string should return empty set."""
    result = find_squares("")
    assert result == set()


def test_short_string():
    """Strings shorter than 4 characters should return empty set."""
    assert find_squares("a") == set()
    assert find_squares("ab") == set()
    assert find_squares("abc") == set()


def test_no_squares():
    """String with no squares should return empty set."""
    result = find_squares("abcdefgh")
    assert result == set()


def test_single_square():
    """String with single square (user example 1)."""
    result = find_squares("abbaabbaBABaa")
    assert result == {"abbaabba"}


def test_multiple_squares():
    """String with multiple squares (user example 2)."""
    result = find_squares("abbaabbaBABaaBaB")
    assert result == {"abbaabba", "aBaB"}


def test_overlapping_squares():
    """Squares at different positions should be deduplicated."""
    result = find_squares("ababababab")
    # Should find "abab" multiple times but return only once
    assert "abab" in result


def test_trivial_squares_excluded():
    """2-character squares should be excluded."""
    result = find_squares("aabbAABB")
    # Should not contain "aa", "bb", "AA", "BB"
    assert "aa" not in result
    assert "bb" not in result
    assert "AA" not in result
    assert "BB" not in result


def test_nested_squares():
    """Square containing another square."""
    result = find_squares("ababababababab")
    # Should find both "abab" and longer squares
    assert "abab" in result
    # Also contains "abababab" = "abab" + "abab"
    assert "abababab" in result


def test_full_word_square():
    """Entire word is a square."""
    result = find_squares("abcdabcd")
    assert "abcdabcd" in result


def test_alternating_pattern():
    """Edge case with alternating pattern."""
    result = find_squares("aBaBaBaB")
    # "aBaB" = "aB" + "aB"
    assert "aBaB" in result
    # Also "aBaBaBaB" = "aBaB" + "aBaB"
    assert "aBaBaBaB" in result


def test_mixed_case_squares():
    """Squares with mixed case letters."""
    result = find_squares("AbaBAbaB")
    # "AbaBAbaB" = "AbaB" + "AbaB"
    assert "AbaBAbaB" in result


def test_random_embedded_squares():
    """Generate random words with known embedded squares and verify detection."""
    random.seed(42)  # For reproducibility

    def random_word(length):
        """Generate random word from alphabet {a, b, A, B}."""
        chars = ['a', 'b', 'A', 'B']
        return ''.join(random.choice(chars) for _ in range(length))

    # Run multiple iterations with different random configurations
    for iteration in range(10):
        # Generate random base words of varying lengths
        base_lengths = [2, 3, 4, 5, 7]
        known_squares = []

        for base_len in base_lengths:
            base = random_word(base_len)
            square = base + base
            known_squares.append(square)

        # Build composite word with embedded squares and random padding
        parts = []
        for i, square in enumerate(known_squares):
            if i > 0:
                # Add random padding between squares
                parts.append(random_word(random.randint(1, 3)))
            parts.append(square)

        # Add prefix and suffix
        word = random_word(random.randint(2, 5)) + ''.join(parts) + random_word(random.randint(2, 5))

        # Find all squares in the composite word
        result = find_squares(word)

        # Verify all known embedded squares are detected
        # (Note: result may contain additional accidental squares from random chars)
        for square in known_squares:
            assert square in result, f"Iteration {iteration}: Expected square '{square}' not found in {result}"
