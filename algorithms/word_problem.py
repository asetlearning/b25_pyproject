# Created by: Alex Myasnikov
# Date: March 9, 2026

import tcgraph_ext as _ext
from algorithms.word_conversion import string_to_word, word_to_string

Word = _ext.Word


def word_from_str(s: str) -> _ext.Word:
    return _ext.Word(string_to_word(s))


def word_to_str(w: _ext.Word) -> str:
    return word_to_string(w.get_letters())


def construct_cayley_approximation(
    word: "_ext.Word | str",
    coset_bound: int = 20_000_000,
) -> "bool | None":
    if isinstance(word, str):
        word = _ext.Word(string_to_word(word))
    return _ext.construct_cayley_approximation(word, coset_bound)

__all__ = ["Word", "word_from_str", "word_to_str", "construct_cayley_approximation"]
