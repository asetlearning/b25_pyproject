# Created by: Alex Myasnikov
# Date: May 14, 2026

"""Phase 1 supplementary script.

Takes (or generates) a list of FactoredWords, trains a small GPT2 LM on the
structured token stream, samples new FactoredWords, writes them as JSON.

Usage:
    python -m experiments.pattern_boost.train_from_factor_words <config.yaml>

See configs/train_from_factor_words.yaml for the schema. The script accepts
the YAML config as its only positional argument (default:
configs/train_from_factor_words.yaml).
"""

import argparse
import json
import logging
import statistics
from pathlib import Path

import yaml

import tcgraph_ext as _ext
from algorithms.logger import configure_logging, log_structured
from algorithms.word_problem import word_to_str
from algorithms.pattern_boost.common import load_challenges, make_scorer
from algorithms.pattern_boost.model import (
    ModelConfig,
    SampleGenConfig,
    TrainConfig,
    build_model,
    pick_device,
    sample as model_sample,
    save_checkpoint,
    train as model_train,
)
from algorithms.pattern_boost.pb_tcgraph import (
    RelatorMap,
    SampleConfig,
    build_standard_relators,
    from_json,
    generate_words,
    to_json,
    uniform_len,
)
from algorithms.pattern_boost.tokenizer import FactoredWordTokenizer

_pb = _ext.pattern_boost

DEFAULT_CONFIG = Path('configs/train_from_factor_words.yaml')


def load_factor_words(path: Path) -> list:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f'expected JSON list in {path}')
    return [from_json(e) for e in entries]


def _summary_stats(values: list) -> dict:
    if not values:
        return {'count': 0}
    sv = sorted(values)

    def pct(p: float):
        k = max(0, min(len(sv) - 1, int(round(p * (len(sv) - 1)))))
        return sv[k]

    return {
        'count': len(values),
        'min': sv[0],
        'max': sv[-1],
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'std': statistics.pstdev(values) if len(values) > 1 else 0.0,
        'p25': pct(0.25),
        'p75': pct(0.75),
        'p95': pct(0.95),
    }


def _print_stats(label: str, s: dict) -> None:
    if s.get('count', 0) == 0:
        print(f"  {label}: (no data)")
        return
    print(f"  {label}: n={s['count']}  min={s['min']:.4g}  max={s['max']:.4g}  "
          f"mean={s['mean']:.4g}  median={s['median']:.4g}  std={s['std']:.4g}  "
          f"p25={s['p25']:.4g}  p75={s['p75']:.4g}  p95={s['p95']:.4g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config', type=Path, nargs='?', default=DEFAULT_CONFIG,
                   help=f'YAML config path (default: {DEFAULT_CONFIG})')
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = cfg['logging'].get('file') or str(output_dir / 'run.jsonl')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=log_file, level=cfg['logging'].get('level', 'info'))
    logger = logging.getLogger('pattern_boost.train')

    log_structured(logger, 'info', event='start',
                   config_path=str(args.config), config=cfg)

    seed = cfg['seed']
    num_generators = cfg['relators']['num_generators']
    max_relator_len = cfg['relators']['max_relator_len']

    # relators required to build tokenizer.
    rm = RelatorMap(build_standard_relators(max_relator_len, num_generators))
    log_structured(logger, 'info', event='relator_map_built', size=len(rm))

    input_path = cfg.get('input')
    generate_initial = cfg.get('generate_initial', 0)
    if input_path:
        fws = load_factor_words(Path(input_path))
        log_structured(logger, 'info', event='factors_loaded', count=len(fws),
                       source=str(input_path))
    else:
        if generate_initial <= 0:
            raise SystemExit('either input or generate_initial > 0 is required in config')
        sc = SampleConfig()
        sc.min_factors = 2
        sc.max_factors = 10
        sc.conj_len = uniform_len(4)
        sc.min_expanded_len = 10
        sc.num_generators = num_generators
        fws = list(generate_words(rm, sc, generate_initial, seed=seed))
        log_structured(logger, 'info', event='factors_generated', count=len(fws))

    tok_cfg = cfg['tokenizer']
    tokenizer = FactoredWordTokenizer(num_generators=num_generators, num_relators=len(rm))
    if tok_cfg.get('use_bpe', False):
        tokenizer.train_bpe(fws, vocab_size=tok_cfg['vocab_size'])
    tokenizer.save(output_dir / 'tokenizer')
    log_structured(logger, 'info', event='tokenizer_ready',
                   use_bpe=bool(tok_cfg.get('use_bpe', False)),
                   vocab_size=tokenizer.vocab_size,
                   base_size=len(tokenizer.base_tokens))

    model_cfg = ModelConfig(
        n_layer=cfg['model']['n_layer'], n_head=cfg['model']['n_head'],
        n_embd=cfg['model']['n_embd'], n_positions=cfg['model']['n_positions'],
    )
    model = build_model(tokenizer, model_cfg)
    device = pick_device()
    log_structured(logger, 'info', event='model_built',
                   device=str(device), vocab_size=tokenizer.vocab_size)

    train_cfg = TrainConfig(
        batch_size=cfg['training']['batch_size'],
        epochs=cfg['training']['epochs'],
        lr=cfg['training']['lr'],
    )
    model_train(model, tokenizer, fws, train_cfg, device=device, logger=logger)

    samp = cfg['sampling']
    sample_cfg = SampleGenConfig(
        n=samp['num_samples_out'], max_length=cfg['model']['n_positions'],
        temperature=samp['temperature'], top_k=samp['top_k'], top_p=samp['top_p'],
        seed=seed, batch_size=cfg['training']['batch_size'],
    )
    sampled = model_sample(model, tokenizer, sample_cfg, device=device)
    log_structured(logger, 'info', event='factors_sampled',
                   requested=samp['num_samples_out'], valid=len(sampled))

    save_checkpoint(model, output_dir / 'model')

    out_path = output_dir / 'sampled_factors.json'
    out_path.write_text(json.dumps([json.loads(to_json(fw, rm)) for fw in sampled], indent=2))
    log_structured(logger, 'info', event='sample_factors_stored', count=len(sampled), file=str(out_path))

    scorer = None
    challenges_path = cfg['scoring'].get('challenges')
    scorer_type = cfg['scoring'].get('type')
    if challenges_path and scorer_type:
        challenge_words, _ = load_challenges(challenges_path)
        scorer = make_scorer(scorer_type, challenge_words, num_generators)
        log_structured(logger, 'info', event='scorer_built',
                       scorer=scorer_type, challenges=len(challenge_words))

    jsonl_path = output_dir / 'sampled_word_score.jsonl'
    scores: list[float] = []
    lengths: list[int] = []
    with jsonl_path.open('w') as f:
        for fw in sampled:
            expanded = rm.expand(fw.factors)
            word_str = word_to_str(expanded)
            entry = {'word': word_str, 'length': len(word_str)}
            lengths.append(len(word_str))
            if scorer is not None:
                s = float(scorer.score_word(expanded))
                entry['score'] = s
                scores.append(s)
            f.write(json.dumps(entry) + '\n')

    log_structured(logger, 'info', event='sample_word_stored',
                   output=str(out_path), jsonl=str(jsonl_path))

    stats = {'length': _summary_stats(lengths)}
    if scores:
        stats['score'] = _summary_stats(scores)
    log_structured(logger, 'info', event='sample_stats', **stats)
    print(f"\nSample stats ({len(sampled)} entries):")
    _print_stats('length', stats['length'])
    if 'score' in stats:
        _print_stats('score', stats['score'])


if __name__ == '__main__':
    main()
