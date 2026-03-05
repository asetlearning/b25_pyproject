"""
Word conversion utilities for Burnside problem data.
Converts between integer arrays and string representations.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

import math
from typing import List, Union


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


def word_inverse(word: str) -> str:
    """Return group-theoretic inverse: reverse the string and swap case (a↔A, b↔B)."""
    return word[::-1].swapcase()


def freely_reduce(word: str) -> str:
    """Freely reduce a word by canceling inverse pairs (aA, Aa, bB, Bb).

    Repeatedly removes adjacent inverse letter pairs until none remain.

    Args:
        word: String over {a, b, A, B}

    Returns:
        Freely reduced string

    Examples:
        >>> freely_reduce("aAbB")
        ''
        >>> freely_reduce("abAb")
        'abAb'
        >>> freely_reduce("abaAb")
        'ab'
    """
    stack = []
    for c in word:
        if stack and stack[-1] == c.swapcase():
            stack.pop()
        else:
            stack.append(c)
    return ''.join(stack)


def cyclically_reduce(word: str) -> str:
    """Cyclically reduce a word by freely reducing and then removing inverse first/last letters.

    A word is cyclically reduced if it is freely reduced and its first and last
    letters are not inverses of each other.

    Args:
        word: String over {a, b, A, B}

    Returns:
        Cyclically reduced string

    Examples:
        >>> cyclically_reduce("aAbB")
        ''
        >>> cyclically_reduce("abBa")
        'b'
        >>> cyclically_reduce("abab")
        'abab'
    """
    word = freely_reduce(word)
    while len(word) >= 2 and word[0] == word[-1].swapcase():
        word = freely_reduce(word[1:-1])
    return word


def word_power(word: str, n: Union[int, float]) -> str:
    """Return the nth power of word (n full or fractional repetitions).

    For integer n: returns word concatenated n times.
    For fractional n: returns the prefix of length n*len(word) from the
    infinite periodic repetition of word.

    Args:
        word: Base word over {a,b,A,B}
        n: Exponent, must be >= 0. If float, n*len(word) must be an integer.

    Returns:
        String of length n*len(word)

    Raises:
        ValueError: if n < 0 or n*len(word) is not an integer

    Examples:
        >>> word_power("ab", 3)
        'ababab'
        >>> word_power("abab", 3.5)
        'ababababababab'
        >>> word_power("ab", 0)
        ''
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    result_len_float = n * len(word)
    result_len = round(result_len_float)
    if abs(result_len_float - result_len) > 1e-9:
        raise ValueError(
            f"n * len(word) must be an integer, got {n} * {len(word)} = {result_len_float}"
        )

    return (word * math.ceil(n))[:result_len]
