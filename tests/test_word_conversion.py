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
    string_to_word
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
