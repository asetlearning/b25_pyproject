# Created by: Alex Myasnikov
# Date: February 26, 2026

from algorithms.rewrite_rules import RewriteRule
from algorithms.word_conversion import word_inverse


def create_power_rewrite_rule(word: str, p: float) -> RewriteRule:
    """
    Build a rewrite rule lhs -> rhs from the decomposition W^5 = W^p * W^q.

    Args:
        word: base word W (string over {a,b,A,B})
        p: exponent, must satisfy 2.5 < p < 5

    Returns:
        RewriteRule where lhs = W^p and rhs = inverse(W^q), q = 5 - p

    Raises:
        ValueError: if p not in (2.5, 5), or p * len(word) is not an integer
    """
    if not (2.5 <= p < 5): # TODO: to accomodate same length powers
        raise ValueError(f"p must satisfy 2.5 < p < 5, got {p}")

    n = len(word)
    lhs_len_float = p * n
    lhs_len = round(lhs_len_float)
    if abs(lhs_len_float - lhs_len) > 1e-9:
        raise ValueError(
            f"p * len(word) must be an integer, got {p} * {n} = {lhs_len_float}"
        )

    w5 = word * 5          # W^5
    w_p = w5[:lhs_len]     # W^p
    w_q = w5[lhs_len:]     # W^q  (length = 5n - lhs_len)

    return RewriteRule(lhs=w_p, rhs=word_inverse(w_q))
