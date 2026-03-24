"""
Word squares detection algorithm.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

from typing import Set

from tqdm import tqdm


def find_squares(word: str, name: str | None = None, show_progress: bool = True) -> Set[str]:
    """
    Find all word squares (W² = WW) in the given word.

    A square is a substring where the first half exactly equals
    the second half. For example, "abbaabba" = "abba" + "abba".

    Args:
        word: Input word string

    Returns:
        Set of all subwords that are squares, including 2-character
        squares (aa, bb, etc.)

    Examples:
        >>> find_squares("abbaabbaBABaa")
        {'abbaabba', 'aa'}
        >>> find_squares("abbaabbaBABaaBaB")
        {'abbaabba', 'aBaB', 'aa'}
    """
    squares = set()

    # Try all possible substrings
    for start in tqdm(range(len(word)), desc=f"{name}: finding squares", disable=not show_progress):
        # For a square, length must be even and >= 2
        for length in range(2, len(word) - start + 1, 2):
            substring = word[start:start + length]
            half = length // 2

            # Check if first half equals second half
            if substring[:half] == substring[half:]:
                squares.add(substring)

    return squares
