# Created by: Alex Myasnikov
# Date: May 14, 2026

"""Tests for FactoredWordTokenizer."""

import tempfile
from pathlib import Path

import pytest

import tcgraph_ext as _ext
from algorithms.pattern_boost.pb_tcgraph import (
    RelatorMap,
    SampleConfig,
    build_standard_relators,
    generate_words,
    uniform_len,
)
from algorithms.pattern_boost.tokenizer import FactoredWordTokenizer

_pb = _ext.pattern_boost


def _make_rm():
    return RelatorMap(build_standard_relators(3))


def _make_samples(n=20, seed=42):
    rm = _make_rm()
    cfg = SampleConfig()
    cfg.min_factors = 1
    cfg.max_factors = 5
    cfg.conj_len = uniform_len(3)
    cfg.min_expanded_len = 1
    return rm, generate_words(rm, cfg, n, seed=seed)


def _fw_equal(a, b) -> bool:
    if len(a.factors) != len(b.factors):
        return False
    for fa, fb in zip(a.factors, b.factors):
        if fa.relator_idx != fb.relator_idx:
            return False
        if list(fa.conjugator.get_letters()) != list(fb.conjugator.get_letters()):
            return False
    return True


def test_base_vocab_layout():
    tok = FactoredWordTokenizer(num_generators=2, num_relators=10)
    # Specials + 2*2 conj letters + 10 relators = 5 + 4 + 10 = 19
    assert len(tok.base_tokens) == 19
    assert tok.base_tokens[:5] == [tok.PAD, tok.BOS, tok.EOS, tok.SEP, tok.CONJ_EMPTY]
    assert '[CONJ_+1]' in tok.base_tokens
    assert '[CONJ_-2]' in tok.base_tokens
    assert '[REL_10]' in tok.base_tokens
    assert tok.pad_token_id != tok.bos_token_id
    assert tok.vocab_size == 19


def test_roundtrip_no_bpe():
    rm, samples = _make_samples(30)
    tok = FactoredWordTokenizer(num_generators=2, num_relators=len(rm))
    for fw in samples:
        ids = tok.encode(fw)
        decoded = tok.decode(ids)
        assert _fw_equal(fw, decoded), 'roundtrip failed (no BPE)'


def test_roundtrip_with_bpe():
    rm, samples = _make_samples(50)
    tok = FactoredWordTokenizer(num_generators=2, num_relators=len(rm))
    tok.train_bpe(samples, vocab_size=128)
    for fw in samples:
        ids = tok.encode(fw)
        decoded = tok.decode(ids)
        assert _fw_equal(fw, decoded), 'roundtrip failed (with BPE)'
    # BPE should compress sequences (post-train avg <= pre-train avg)
    pre_lens = []
    post_lens = []
    pre_tok = FactoredWordTokenizer(num_generators=2, num_relators=len(rm))
    for fw in samples:
        pre_lens.append(len(pre_tok.encode(fw)))
        post_lens.append(len(tok.encode(fw)))
    assert sum(post_lens) <= sum(pre_lens), 'BPE did not compress sequences'


def test_empty_conjugator():
    rm = _make_rm()
    tok = FactoredWordTokenizer(num_generators=2, num_relators=len(rm))
    fw = _pb.FactoredWord()
    f = _pb.Factor()
    f.relator_idx = 1
    f.conjugator = _ext.Word([])
    fw.factors = [f]
    ids = tok.encode(fw)
    decoded = tok.decode(ids)
    assert _fw_equal(fw, decoded)
    base_tokens = tok._ids_to_base_tokens(ids)
    assert tok.CONJ_EMPTY in base_tokens


def test_save_load_no_bpe(tmp_path):
    rm, samples = _make_samples(10)
    tok = FactoredWordTokenizer(num_generators=2, num_relators=len(rm))
    tok.save(tmp_path / 'tok')
    loaded = FactoredWordTokenizer.load(tmp_path / 'tok')
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.pad_token_id == tok.pad_token_id
    for fw in samples:
        assert tok.encode(fw) == loaded.encode(fw)
        assert _fw_equal(fw, loaded.decode(loaded.encode(fw)))


def test_save_load_with_bpe(tmp_path):
    rm, samples = _make_samples(30)
    tok = FactoredWordTokenizer(num_generators=2, num_relators=len(rm))
    tok.train_bpe(samples, vocab_size=80)
    tok.save(tmp_path / 'tok')
    loaded = FactoredWordTokenizer.load(tmp_path / 'tok')
    assert loaded.vocab_size == tok.vocab_size
    for fw in samples:
        assert tok.encode(fw) == loaded.encode(fw)
        assert _fw_equal(fw, loaded.decode(loaded.encode(fw)))


def test_try_decode_returns_none_on_garbled():
    rm = _make_rm()
    tok = FactoredWordTokenizer(num_generators=2, num_relators=len(rm))
    # SEP without REL — malformed
    bad = [tok.bos_token_id, tok.sep_token_id, tok.eos_token_id]
    assert tok.try_decode(bad) is None


def test_try_decode_returns_none_on_empty_factors():
    rm = _make_rm()
    tok = FactoredWordTokenizer(num_generators=2, num_relators=len(rm))
    # BOS EOS only — no factors
    assert tok.try_decode([tok.bos_token_id, tok.eos_token_id]) is None


def test_rejects_out_of_range_relator_idx():
    rm = _make_rm()
    tok = FactoredWordTokenizer(num_generators=2, num_relators=5)
    fw = _pb.FactoredWord()
    f = _pb.Factor()
    f.relator_idx = 999  # out of range
    f.conjugator = _ext.Word([])
    fw.factors = [f]
    with pytest.raises(ValueError):
        tok.encode(fw)


def test_vocab_size_too_small_raises():
    tok = FactoredWordTokenizer(num_generators=2, num_relators=10)
    with pytest.raises(ValueError):
        tok.train_bpe([], vocab_size=5)
