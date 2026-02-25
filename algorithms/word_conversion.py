"""
Word conversion utilities for Burnside problem data.
Converts between integer arrays and string representations.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

from typing import List


def int_to_char(n: int) -> str:
    """Convert single integer to character.

    Args:
        n: Integer in {-2, -1, 1, 2}

    Returns:
        Character: 1->a, 2->b, -1->A, -2->B

    Raises:
        ValueError: If n not in {-2, -1, 1, 2}
    """
    mapping = {1: 'a', 2: 'b', -1: 'A', -2: 'B'}
    if n not in mapping:
        raise ValueError(f"Invalid integer: {n}. Must be in {{-2, -1, 1, 2}}")
    return mapping[n]


def word_to_string(word: List[int]) -> str:
    """Convert integer array to string.

    Args:
        word: List of integers in {-2, -1, 1, 2}

    Returns:
        String representation

    Raises:
        ValueError: If any element invalid
    """
    return ''.join(int_to_char(n) for n in word)


def char_to_int(c: str) -> int:
    """Convert character to integer.

    Args:
        c: Character in {a, b, A, B}

    Returns:
        Integer: a->1, b->2, A->-1, B->-2

    Raises:
        ValueError: If c not in {a, b, A, B}
    """
    mapping = {'a': 1, 'b': 2, 'A': -1, 'B': -2}
    if c not in mapping:
        raise ValueError(f"Invalid character: {c}. Must be in {{a, b, A, B}}")
    return mapping[c]


def string_to_word(s: str) -> List[int]:
    """Convert string to integer array.

    Args:
        s: String of characters in {a, b, A, B}

    Returns:
        List of integers

    Raises:
        ValueError: If any character invalid
    """
    return [char_to_int(c) for c in s]
