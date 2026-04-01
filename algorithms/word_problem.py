# Created by: Alex Myasnikov
# Date: March 9, 2026

import tcgraph_ext as _ext
from algorithms.word_conversion import string_to_word, word_to_string

Word = _ext.Word


def word_from_str(s: str) -> _ext.Word:
    return _ext.Word(string_to_word(s))


def word_to_str(w: _ext.Word) -> str:
    return word_to_string(w.get_letters())


def attempt_to_reduce_using_relations(
    word: "_ext.Word | str",
    relators: "list[_ext.Word | str]",
) -> "bool | None":
    if isinstance(word, str):
        word = _ext.Word(string_to_word(word))
    relators = [
        _ext.Word(string_to_word(r)) if isinstance(r, str) else r
        for r in relators
    ]
    return _ext.attempt_to_reduce_using_relations(word, relators)


def attempt_to_reduce_using_relations_greedy(
    word: "_ext.Word | str",
    relators: "list[_ext.Word | str]",
    word_name: str = "R",
    check_point_file_name: "str | None" = None,
) -> "bool | None":
    if isinstance(word, str):
        word = _ext.Word(string_to_word(word))
    relators = [
        _ext.Word(string_to_word(r)) if isinstance(r, str) else r
        for r in relators
    ]
    return _ext.attempt_to_reduce_using_relations_greedy(word, relators, word_name, check_point_file_name)


def attempt_to_decide_if_trivial(
    word: "_ext.Word | str",
    coset_bound: int = 20_000_000,
    relator_max_length: int = 5,
    log_step: int = 10_000_000,
    len_log_step: int = 100_000_000,
) -> "bool | None":
    if isinstance(word, str):
        word = _ext.Word(string_to_word(word))
    return _ext.attempt_to_decide_if_trivial(word, coset_bound, relator_max_length, log_step, len_log_step)


def greedy_reduce2_5(word: "_ext.Word | str") -> "_ext.Word":
    if isinstance(word, str):
        word = _ext.Word(string_to_word(word))
    # Have to return string because Word is not pickable by default
    return word_to_str(_ext.greedy_reduce2_5(word))


def greedy_reduce_cyclic2_5(word: "_ext.Word | str") -> "_ext.Word":
    if isinstance(word, str):
        word = _ext.Word(string_to_word(word))
    # Have to return string because Word is not pickable by default
    return word_to_str(_ext.greedy_reduce_cyclic2_5(word))


def greedy_reduce_cyclic_exhaustive2_5(word: "_ext.Word | str") -> "_ext.Word":
    if isinstance(word, str):
        word = _ext.Word(string_to_word(word))
    # Have to return string because Word is not pickable by default
    return word_to_str(_ext.greedy_reduce_cyclic_exhaustive2_5(word))


__all__ = [
    "Word", "word_from_str", "word_to_str",
    "attempt_to_reduce_using_relations",
    "attempt_to_reduce_using_relations_greedy",
    "attempt_to_decide_if_trivial",
    "greedy_reduce2_5",
    "greedy_reduce_cyclic2_5",
    "greedy_reduce_cyclic_exhaustive2_5",
]
