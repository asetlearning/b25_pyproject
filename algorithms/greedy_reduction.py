# Created by: Alex Myasnikov
# Date: February 26, 2026
from tqdm import tqdm

from algorithms.word_conversion import word_inverse
from algorithms.word_squares import find_squares
from algorithms.maximal_power import find_maximal_powers
from algorithms.power_rules import create_power_rewrite_rule
from algorithms.rewrite_rules import apply_rule
from algorithms.logger import get_logger, log_structured

logger = get_logger(__name__)

_VALID_CHARS = {'a', 'b', 'A', 'B'}


def greedy_reduce(word: str, name: str) -> str:
    """Repeatedly find and reduce the highest-exponent power subword.

    Iterates until no power with exponent > 2.5 remains:
      1. Find all squares (WW subwords)
      2. Extract bases from squares
      3. Find maximal power for each base
      4. Apply power rewrite rule on highest-exponent base

    Args:
        word: Input word over {a,b,A,B}

    Returns:
        Reduced word (same or shorter)
    """
    assert word, "word must be non-empty"
    assert all(c in _VALID_CHARS for c in word), f"Invalid chars in word: {word}"

    current = word
    i = 0

    while True:
        squares = find_squares(current, name)
        if not squares:
            break

        log_structured(
            logger, 'info',
            event='greedy_reduction_squares',
            relator=name,
            iteration=i,
            n_squares=len(squares),
        )

        bases = {sq[:len(sq) // 2] for sq in squares}

        best_base = None
        best_power = 0.0
        best_score = 0

        for base in tqdm(bases, desc=f"{name}: finding maximal powers"):
            base_len = len(base)
            results = find_maximal_powers(base, current)
            for _, power_count, start_pos in results:
                # the score is len(lhs) - len(rhs) i.e. how much reduction will the rule give
                the_score = power_count * base_len - (5-power_count) * base_len
                # if power_count > best_power:  # sumple measure with the highest power
                if the_score > best_score:
                    best_power = power_count
                    best_base = base
                    best_pos = start_pos
                    best_score = the_score

        if best_power <= 2.5:
            break

        log_structured(
            logger, 'info',
            event='greedy_reduction_maximal_powers',
            relator=name,
            iteration=i,
            largest_exponent=best_power,
        )

        # Normalize to (2.5, 5): W^p = W^(p mod 5) since W^5 = e
        effective_power = best_power
        assert effective_power < 5, f"Effective power is greater then 5: {effective_power}"
        # if effective_power >= 5.0:
        #     effective_power = best_power % 5.0
        #     if effective_power == 0.0 or effective_power <= 2.5:
        #         break

        rule = create_power_rewrite_rule(best_base, effective_power)
        assert rule.lhs + word_inverse(rule.rhs) == best_base*5, f"Rule {rule} is incorrect"
        assert rule.lhs in current, f"Rule {rule} does not apply to word {current}"

        new_word, changed = apply_rule(current, rule)

        assert len(new_word) < len(current), f"Rule {rule} did not shorten word"
        if len(new_word) == len(current):
            break

        log_structured(
            logger, 'info',
            event='greedy_reduction_step',
            iteration=i,
            relator=name,
            original_length=len(current),
            new_length=len(new_word),
            reduction=len(current) - len(new_word),
            base=best_base,
            exponent=best_power,
            rule=str(rule),
        )

        current = new_word
        i += 1

    return current
