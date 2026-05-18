# Created by: Alex Myasnikov
# Date: May 14, 2026

"""Smoke tests for PatternBoost model factory + train/sample."""

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from algorithms.pattern_boost.pb_tcgraph import (
    RelatorMap,
    SampleConfig,
    build_standard_relators,
    generate_words,
    uniform_len,
)
from algorithms.pattern_boost.tokenizer import FactoredWordTokenizer
from algorithms.pattern_boost.model import (
    ModelConfig,
    TrainConfig,
    SampleGenConfig,
    build_model,
    train,
    sample,
    save_checkpoint,
    load_checkpoint,
)


def _setup(n_samples=16):
    rm = RelatorMap(build_standard_relators(3))
    cfg = SampleConfig()
    cfg.min_factors = 1
    cfg.max_factors = 3
    cfg.conj_len = uniform_len(2)
    cfg.min_expanded_len = 1
    samples = generate_words(rm, cfg, n_samples, seed=7)
    tok = FactoredWordTokenizer(num_generators=2, num_relators=len(rm))
    return rm, samples, tok


def test_build_model_vocab_matches_tokenizer():
    _, _, tok = _setup()
    model = build_model(tok, ModelConfig(n_layer=1, n_head=2, n_embd=32, n_positions=64))
    assert model.config.vocab_size == tok.vocab_size
    assert model.config.bos_token_id == tok.bos_token_id


def test_train_and_sample_smoke():
    _, samples, tok = _setup()
    model = build_model(tok, ModelConfig(n_layer=1, n_head=2, n_embd=32, n_positions=64))
    train(model, tok, samples, TrainConfig(batch_size=4, epochs=1, log_every=10000),
          device=torch.device('cpu'))
    fws = sample(model, tok, SampleGenConfig(n=4, max_length=64, batch_size=4, seed=1),
                 device=torch.device('cpu'))
    # Not all generations need to be valid, but list must be the list type
    assert isinstance(fws, list)


def test_checkpoint_save_load(tmp_path):
    _, _, tok = _setup()
    model = build_model(tok, ModelConfig(n_layer=1, n_head=2, n_embd=32, n_positions=64))
    save_checkpoint(model, tmp_path / 'ckpt')
    loaded = load_checkpoint(tmp_path / 'ckpt')
    assert loaded.config.vocab_size == model.config.vocab_size
