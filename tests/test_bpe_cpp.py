# Created by: Alex Myasnikov
# Date: June 3, 2026

"""Tests verifying C++ BPETokenizer/WordCompressor match their Python counterparts."""

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


# ---------- WordCompressor tests ----------

@pytest.fixture
def py_tok_and_cpp_wc():
    """Trained BPETokenizerWordCompression + matching C++ WordCompressor."""
    py_tok = _make_trained_tokenizer()
    cpp_wc = _pb.WordCompressor(bpe_model_from_compressor(py_tok))
    return py_tok, cpp_wc


def test_word_compressor_num_generators(py_tok_and_cpp_wc):
    py_tok, cpp_wc = py_tok_and_cpp_wc
    eow_id = py_tok.inverse_vocab[py_tok.eow]
    expected = len([tid for tid in py_tok.vocab if tid != eow_id])
    assert cpp_wc.num_generators == expected


def test_word_compressor_encode_str(py_tok_and_cpp_wc):
    py_tok, cpp_wc = py_tok_and_cpp_wc
    eow_id = py_tok.inverse_vocab[py_tok.eow]
    non_eow_ids = sorted(tid for tid in py_tok.vocab if tid != eow_id)
    token_to_gen = {tid: i + 1 for i, tid in enumerate(non_eow_ids)}
    for word_str in _TEST_WORDS:
        expected_gens = [token_to_gen[tid] for tid in py_tok.tokenize_with_bpe(word_str)]
        cpp_encoded = cpp_wc.encode_str(word_str)
        assert list(cpp_encoded.get_letters()) == expected_gens, \
            f"encode_str mismatch for '{word_str}'"


def test_word_compressor_decode_word(py_tok_and_cpp_wc):
    _, cpp_wc = py_tok_and_cpp_wc
    for word_str in _TEST_WORDS:
        compressed = cpp_wc.encode_str(word_str)
        decoded = word_to_str(cpp_wc.decode_word(compressed))
        assert decoded == word_str, \
            f"decode_word roundtrip failed for '{word_str}': got '{decoded}'"


def test_word_compressor_compressed_repr(py_tok_and_cpp_wc):
    from algorithms.word_conversion import word_inverse
    py_tok, cpp_wc = py_tok_and_cpp_wc
    eow_id = py_tok.inverse_vocab[py_tok.eow]
    non_eow_ids = sorted(tid for tid in py_tok.vocab if tid != eow_id)
    token_to_gen = {tid: i + 1 for i, tid in enumerate(non_eow_ids)}
    gen_to_token = {g: tid for tid, g in token_to_gen.items()}
    for word_str in _TEST_WORDS:
        compressed = cpp_wc.encode_str(word_str)
        parts = []
        for k in compressed.get_letters():
            tok_str = py_tok.vocab[gen_to_token[abs(k)]]
            parts.append(f"[{tok_str if k > 0 else word_inverse(tok_str)}]")
        assert cpp_wc.compressed_repr(compressed) == "".join(parts), \
            f"compressed_repr mismatch for '{word_str}'"


def test_word_compressor_roundtrip(py_tok_and_cpp_wc):
    _, cpp_wc = py_tok_and_cpp_wc
    for word_str in _TEST_WORDS:
        compressed = cpp_wc.encode_str(word_str)
        decoded = word_to_str(cpp_wc.decode_word(compressed))
        assert decoded == word_str, \
            f"roundtrip failed for '{word_str}': got '{decoded}'"
