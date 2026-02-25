"""
Maximal power detection algorithm for repeated subwords.

Finds maximal powers of a subword S within a word W, where a power
is consecutive repetitions (full or fractional) of S.

Created by: Alex Myasnikov
Date: February 12, 2026
"""

from typing import List, Tuple


def find_maximal_powers(subword: str, word: str) -> List[Tuple[str, float, int]]:
    """Find maximal powers of subword in word.

    A "power" is consecutive repetitions of subword (full or fractional).
    "Maximal" means cannot extend before/after. Only powers > 1 included.

    Args:
        subword: Pattern to find repetitions of (from alphabet {a,b,A,B})
        word: Word to search in (from alphabet {a,b,A,B})

    Returns:
        List of (substring, power_count, start_position) tuples where:
        - substring: maximal power extracted from word
        - power_count: number of repetitions (e.g., 2.75)
        - start_position: index in word where power starts

    Raises:
        ValueError: If subword or word contains invalid characters

    Examples:
        >>> find_maximal_powers("aaBB", "bbaaBBaaBBaaBAAaa")
        [("aaBBaaBBaaB", 2.75, 2)]

        >>> find_maximal_powers("ab", "ababababAabab")
        [("abababab", 4.0, 0), ("abab", 2.0, 9)]
    """
    # Validate inputs
    if not subword or not word:
        return []

    if len(subword) > len(word):
        return []

    valid_chars = {'a', 'b', 'A', 'B'}
    for c in subword:
        if c not in valid_chars:
            raise ValueError(f"Invalid character in subword: {c}. Must be in {{a, b, A, B}}")
    for c in word:
        if c not in valid_chars:
            raise ValueError(f"Invalid character in word: {c}. Must be in {{a, b, A, B}}")

    results = []
    covered_until = 0
    sub_len = len(subword)

    # Scan word left-to-right
    i = 0
    while i < len(word):
        # Skip if already covered by previous maximal power
        if i < covered_until:
            i += 1
            continue

        # Check if subword matches at position i
        if word[i:i + sub_len] != subword:
            i += 1
            continue

        # Found subword at position i - compute maximal power
        end_pos, power_count = _compute_power_at_position(subword, word, i)

        # Only include if power > 1
        if power_count > 1.0:
            substring = word[i:end_pos]
            results.append((substring, power_count, i))
            covered_until = end_pos

        i += 1

    return results


def _compute_power_at_position(subword: str, word: str, start: int) -> Tuple[int, float]:
    """Compute power of subword starting at given position.

    Args:
        subword: Pattern being repeated
        word: Word containing the pattern
        start: Starting position in word

    Returns:
        Tuple of (end_position, power_count) where:
        - end_position: index where power ends (exclusive)
        - power_count: total repetitions including fractional part
    """
    sub_len = len(subword)
    pos = start
    full_reps = 0

    # Count full repetitions
    while pos + sub_len <= len(word) and word[pos:pos + sub_len] == subword:
        full_reps += 1
        pos += sub_len

    # Check for fractional part (prefix of subword)
    fractional = 0.0
    remaining_len = len(word) - pos
    if remaining_len > 0:
        # Find longest matching prefix
        match_len = 0
        for j in range(min(remaining_len, sub_len)):
            if word[pos + j] == subword[j]:
                match_len += 1
            else:
                break

        if match_len > 0:
            fractional = match_len / sub_len
            pos += match_len

    total_power = full_reps + fractional
    return pos, total_power
