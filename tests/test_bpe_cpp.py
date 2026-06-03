# Created by: Alex Myasnikov
# Date: June 3, 2026

"""Tests verifying C++ BPETokenizer produces identical results to Python BPETokenizerWordCompression."""

import pytest

import tcgraph_ext as _ext
from algorithms.word_compression import BPETokenizerWordCompression
from algorithms.word_problem import word_from_str, word_to_str
from algorithms.pattern_boost.pb_tcgraph import bpe_model_from_compressor

_pb = _ext.pattern_boost

_ALPHABET = ['a', 'b', 'A', 'B']
_TEST_WORDS = ["a", "b", "ab", "abab", "ababAB", "aAbBaAbB", "bBaA", "aabb", "AABB"]


def _make_trained_tokenizer(vocab_size: int = 20) -> BPETokenizerWordCompression:
    words = ["abababababab", "ababABABABab", "aAbBaAbBaAbB", "bBaAbBaAbBaA", "aabbAABB"]
    tok = BPETokenizerWordCompression(_ALPHABET)
    tok.train(words, vocab_size=vocab_size)
    return tok


def test_bpe_model_construction():
    py_tok = _make_trained_tokenizer()
    model = bpe_model_from_compressor(py_tok)
    assert len(model.vocab) == len(py_tok.vocab)
    assert len(model.merges) == len(py_tok.bpe_merges)


def test_encode_matches_python():
    py_tok = _make_trained_tokenizer()
    cpp_tok = _pb.BPETokenizer(bpe_model_from_compressor(py_tok))
    for word_str in _TEST_WORDS:
        word = word_from_str(word_str)
        cpp_ids = cpp_tok.encode(word)
        py_ids = py_tok.tokenize_with_bpe(word_str)
        assert cpp_ids == py_ids, f"encode mismatch for '{word_str}': cpp={cpp_ids} py={py_ids}"


def test_decode_matches_python():
    py_tok = _make_trained_tokenizer()
    cpp_tok = _pb.BPETokenizer(bpe_model_from_compressor(py_tok))
    for word_str in _TEST_WORDS:
        ids = py_tok.tokenize_with_bpe(word_str)
        cpp_decoded = word_to_str(cpp_tok.decode(ids))
        py_decoded = py_tok.decode(ids)
        assert cpp_decoded == py_decoded, f"decode mismatch for ids {ids}: cpp='{cpp_decoded}' py='{py_decoded}'"


def test_encode_decode_roundtrip():
    py_tok = _make_trained_tokenizer()
    cpp_tok = _pb.BPETokenizer(bpe_model_from_compressor(py_tok))
    for word_str in _TEST_WORDS:
        word = word_from_str(word_str)
        ids = cpp_tok.encode(word)
        decoded = word_to_str(cpp_tok.decode(ids))
        assert decoded == word_str, f"roundtrip failed for '{word_str}': got '{decoded}'"


def test_encode_empty_word():
    py_tok = _make_trained_tokenizer()
    cpp_tok = _pb.BPETokenizer(bpe_model_from_compressor(py_tok))
    assert cpp_tok.encode(_ext.Word([])) == []


def test_no_applicable_merges():
    # Tokenizer with no merges trained — each char maps to its base token ID
    tok = BPETokenizerWordCompression(_ALPHABET)
    tok.train([], vocab_size=len(_ALPHABET) + 1)  # vocab_size <= base vocab size → no merges
    cpp_tok = _pb.BPETokenizer(bpe_model_from_compressor(tok))
    for word_str in ["a", "b", "A", "B", "ab"]:
        word = word_from_str(word_str)
        cpp_ids = cpp_tok.encode(word)
        py_ids = tok.tokenize_with_bpe(word_str)
        assert cpp_ids == py_ids, f"no-merge mismatch for '{word_str}'"
