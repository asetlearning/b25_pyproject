# Created by: Alex Myasnikov
# Date: February 26, 2026
from tqdm import tqdm

from algorithms.word_conversion import (
    word_inverse,
    minimal_period,
    word_power,
    freely_reduce,
    cyclically_permute,
    cyclically_reduce,
)
from algorithms.word_squares import find_squares
from algorithms.maximal_power import find_maximal_powers
from algorithms.power_rules import create_power_rewrite_rule
from algorithms.rewrite_rules import RewriteRule, apply_rule
from algorithms.logger import get_logger, log_structured

logger = get_logger(__name__)

_VALID_CHARS = {'a', 'b', 'A', 'B'}


def _greedy_reduce(word: str, name: str, show_progress: bool) -> str:
    """Core greedy reduction loop. Assumes word is already cyclically reduced."""
    current = word
    explored_words = {current}
    i = 0

    while True:
        squares = find_squares(current, name, show_progress=show_progress)
        if not squares:
            break

        log_structured(
            logger, 'info',
            event='greedy_reduction_squares',
            relator=name,
            iteration=i,
            n_squares=len(squares),
        )

        bases = {minimal_period(sq[:len(sq) // 2]) for sq in squares}

        best_base = None
        best_power = 0.0
        best_score = -1.0 # TODO: to accommodate same length

        for base in tqdm(bases, desc=f"{name}: finding maximal powers", disable=not show_progress):
            base_len = len(base)
            results = find_maximal_powers(base, current)
            for _, power_count, start_pos in results:
                # the score is len(lhs) - len(rhs) i.e. how much reduction will the rule give
                the_score = power_count * base_len - (5 - power_count) * base_len
                # if power_count > best_power:  # simple measure with the highest power
                if the_score > best_score:
                    best_power = power_count
                    best_base = base
                    best_pos = start_pos
                    best_score = the_score

#        if best_power <= 2.5: TODO: experiment use words with the same length
        if best_power < 2.5:
            break

        log_structured(
            logger, 'info',
            event='greedy_reduction_maximal_powers',
            relator=name,
            iteration=i,
            largest_exponent=best_power,
        )

        # Handle power >= 5: W^5 = identity
        q = best_power % 5.0
        n = int(best_power) // 5
        if n >= 1:
            # W^(5n) -> ""; loop reduces 5 copies per iteration; q handled in next iteration
            rule = RewriteRule(lhs=word_power(best_base, 5), rhs="")
        elif q >= 2.5:  # TODO: allow same length words
            rule = create_power_rewrite_rule(best_base, q)
        else:
            break  # no useful rule

        assert rule.lhs + word_inverse(rule.rhs) == best_base * 5, f"Rule {rule} is incorrect"
        assert rule.lhs in current, f"Rule {rule} does not apply to word {current}"

        new_word, changed = apply_rule(current, rule)

        # TODO: allow same length words
        assert len(new_word) <= len(current), f"Rule {rule} increased word {current}"
        # if len(new_word) == len(current):
        if not changed or new_word in explored_words:
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
        if current == "":
            break

        explored_words.add(current)
        i += 1

    return current


def greedy_reduce(word: str, name: str, show_progress: bool = True) -> str:
    """Cyclically reduce then apply greedy power-rewrite until no exponent > 2.5 remains.

    Args:
        word: Input word over {a,b,A,B}

    Returns:
        Reduced word (same or shorter)
    """
    assert word, "word must be non-empty"
    assert all(c in _VALID_CHARS for c in word), f"Invalid chars in word: {word}"

    word = cyclically_reduce(word)
    if not word:
        return ""
    return _greedy_reduce(word, name, show_progress)


def greedy_reduce_cyclic(word: str, name: str, show_progress: bool = True) -> str:
    """Apply greedy reduction to every cyclic permutation; return minimal-length result.

    Cyclically reduces once, then iterates all len(word) permutations of the result.
    Freely reduces each permutation (boundary cancellation), then runs _greedy_reduce.
    Returns "" as soon as any permutation is trivial.

    Args:
        word: Input word over {a,b,A,B}

    Returns:
        Shortest reduced word across all permutations, or "" if any is trivial
    """
    assert word, "word must be non-empty"
    assert all(c in _VALID_CHARS for c in word), f"Invalid chars in word: {word}"

    word = cyclically_reduce(word)
    if not word:
        return ""

    n = len(word)
    best: str | None = None

    for i in range(n):
        perm = freely_reduce(cyclically_permute(word, i))

        log_structured(
            logger, 'info',
            event='cyclic_permutation_attempt',
            relator=name,
            permutation=i,
            perm_length=len(perm),
        )

        if not perm:
            log_structured(logger, 'info', event='cyclic_permutation_trivial', relator=name, permutation=i)
            return ""

        result = _greedy_reduce(perm, f"{name}[cyc={i}]", show_progress)

        log_structured(
            logger, 'info',
            event='cyclic_permutation_result',
            relator=name,
            permutation=i,
            original_length=len(perm),
            result_length=len(result),
        )

        if not result:
            return ""

        if best is None or len(result) < len(best):
            best = result

    return best  # type: ignore[return-value]  # best is set since n >= 1


def greedy_reduce_cyclic_exhaustive(word: str, name: str, show_progress: bool = True) -> str:
    """Repeatedly apply greedy_reduce_cyclic until no further reduction.

    Args:
        word: Input word over {a,b,A,B}

    Returns:
        Shortest word found, or "" if trivial
    """
    assert word, "word must be non-empty"
    assert all(c in _VALID_CHARS for c in word), f"Invalid chars in word: {word}"

    current = word
    iteration = 0

    while True:
        result = greedy_reduce_cyclic(current, f"{name}[ex={iteration}]", show_progress)

        log_structured(
            logger, 'info',
            event='cyclic_exhaustive_step',
            relator=name,
            iteration=iteration,
            input_length=len(current),
            result_length=len(result),
        )

        if not result or len(result) >= len(current):
            return result

        current = result
        iteration += 1
