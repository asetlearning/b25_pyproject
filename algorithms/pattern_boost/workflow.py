# Created by: Alex Myasnikov
# Date: May 14, 2026

"""PatternBoost main workflow loop.

Implements the train -> sample -> local-search -> score -> keep-top loop
described in `Pattern boost implementation details.md`.

Config is a nested dataclass loaded from YAML via `load_config()`.
Run via `run_pattern_boost(cfg)`.
"""

import gc
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml
from tqdm import tqdm

import tcgraph_ext as _ext
from algorithms.logger import configure_logging, log_structured
from algorithms.word_problem import word_to_str, greedy_reduce2_5
from algorithms.pattern_boost.common import load_challenges, make_scorer
from algorithms.pattern_boost.model import (
    ModelConfig,
    SampleGenConfig,
    TrainConfig,
    build_model,
    load_checkpoint,
    pick_device,
    sample as model_sample,
    save_checkpoint,
    train as model_train,
)
from algorithms.pattern_boost.pb_tcgraph import (
    LocalSearchConfig,
    LocalSearcher,
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


# ---------- config schema ----------

@dataclass
class RelatorsCfg:
    max_relator_len: int = 5
    num_generators: int = 2


@dataclass
class ChallengesCfg:
    path: str = ''
    sample_size: Optional[int] = None  # if set, load a random subset of this many


@dataclass
class TokenizerCfg:
    use_bpe: bool = False
    vocab_size: int = 512  # only used when use_bpe=True


@dataclass
class InitialSamplingCfg:
    n_initial: int = 1000
    min_factors: int = 2
    max_factors: int = 10
    min_expanded_len: int = 10
    conj_len_max: int = 4


@dataclass
class GenerationCfg:
    n_samples: int = 200
    max_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    batch_size: int = 16


@dataclass
class LocalSearchCfg:
    beam_width: int = 50
    max_steps: int = 100
    num_conj_samples: int = 5
    num_insert_samples: int = 5
    conj_max_len: int = 4
    min_factors: int = 1
    max_factors: int = 50


@dataclass
class ScoringCfg:
    type: str = 'hellinger'  # hellinger | hamming | edit


@dataclass
class WorkflowLoopCfg:
    num_iterations: int = 10
    top_k: int = 100
    max_pool_size: int = 5000
    parallel_workers: int = 4
    save_every: int = 1
    train_from_scratch: bool = False  # rebuild model with fresh weights each iter


@dataclass
class LoggingCfg:
    level: str = 'info'
    log_file: Optional[str] = None


@dataclass
class WorkflowConfig:
    output_dir: str = 'output/pattern_boost'
    seed: int = 42
    init_model_from: Optional[str] = None  # path to saved model checkpoint dir
    relators: RelatorsCfg = field(default_factory=RelatorsCfg)
    challenges: ChallengesCfg = field(default_factory=ChallengesCfg)
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    initial_sampling: InitialSamplingCfg = field(default_factory=InitialSamplingCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    local_search: LocalSearchCfg = field(default_factory=LocalSearchCfg)
    scoring: ScoringCfg = field(default_factory=ScoringCfg)
    workflow: WorkflowLoopCfg = field(default_factory=WorkflowLoopCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)


def _build_nested(dc_cls, src):
    if src is None:
        return dc_cls()
    fields_map = {f.name: f for f in dc_cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in src.items():
        if k not in fields_map:
            raise ValueError(f'unknown config key {k!r} for {dc_cls.__name__}')
        ftype = fields_map[k].type
        # If the field type is itself a dataclass-like name, recurse
        if isinstance(v, dict) and hasattr(ftype, '__dataclass_fields__'):
            kwargs[k] = _build_nested(ftype, v)
        else:
            kwargs[k] = v
    return dc_cls(**kwargs)


def load_config(path) -> WorkflowConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return _build_nested(WorkflowConfig, raw or {})


# ---------- helpers ----------

def _make_local_search_cfg(cfg: LocalSearchCfg, num_generators: int) -> LocalSearchConfig:
    ls = LocalSearchConfig()
    ls.beam_width = cfg.beam_width
    ls.max_steps = cfg.max_steps
    ls.num_conj_samples = cfg.num_conj_samples
    ls.num_insert_samples = cfg.num_insert_samples
    ls.conj_max_len = cfg.conj_max_len
    ls.min_factors = cfg.min_factors
    ls.max_factors = cfg.max_factors
    ls.num_generators = num_generators
    return ls


def _make_sample_cfg(cfg: InitialSamplingCfg, num_generators: int) -> SampleConfig:
    sc = SampleConfig()
    sc.min_factors = cfg.min_factors
    sc.max_factors = cfg.max_factors
    sc.conj_len = uniform_len(cfg.conj_len_max)
    sc.min_expanded_len = cfg.min_expanded_len
    sc.num_generators = num_generators
    return sc


def _dedup_key(fw, rm: RelatorMap) -> str:
    return to_json(fw, rm)


def _merge_pool(pool: list, new_fws: list, rm: RelatorMap, max_pool_size: int) -> list:
    seen = {_dedup_key(fw, rm) for fw in pool}
    merged = list(pool)
    for fw in new_fws:
        k = _dedup_key(fw, rm)
        if k not in seen:
            seen.add(k)
            merged.append(fw)
    if len(merged) > max_pool_size:
        merged = merged[-max_pool_size:]
    return merged


def _score_summary(scores: list[float]) -> dict:
    if not scores:
        return {'count': 0}
    return {
        'count': len(scores),
        'min': min(scores),
        'max': max(scores),
        'mean': statistics.mean(scores),
        'median': statistics.median(scores),
    }


def _serialize_pool(fws: list, rm: RelatorMap) -> list:
    return [json.loads(to_json(fw, rm)) for fw in fws]


def _release_model(model) -> None:
    """Move model off-device, drop refs, force GC, empty device allocator caches."""
    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available() and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


# ---------- stopping criteria ----------

def check_stopping_criteria(
    results: list,
    rm: RelatorMap,
    challenge_strs: dict,
    logger,
    iteration: int,
    matches_path: Optional[Path] = None,
    scorer_type: str = '',
    perfect_score: float = 1.0,
    tol: float = 1e-6,
) -> bool:
    """Apply per-iteration stopping criteria after local search.

    Criteria:
      1. Log any candidate whose score is (numerically) `perfect_score` —
         informational, does NOT terminate.
      2. Log any candidate whose expanded B(2,5) word is cyclically equal to
         any challenge string. **Terminates** the iteration loop.

    Returns True iff the iteration loop should stop early.
    """
    terminate = False

    for score, fw in results:
        if abs(float(score) - perfect_score) < tol:
            expanded_str = word_to_str(rm.expand(fw.factors))
            log_structured(logger, 'info', event='pb_perfect_score_candidate',
                           iteration=iteration, score=float(score),
                           word=expanded_str)

    cand_lengths = []
    cand_red_lengths = []
    for score, fw in results:
        cand_str = word_to_str(rm.expand(fw.factors))
        if not cand_str:
            continue

        cand_red = greedy_reduce2_5(cand_str)
        cand_lengths.append(len(cand_str))
        cand_red_lengths.append(len(cand_red))

        doubled = cand_str + cand_str
        for name, chal_str in challenge_strs.items():
            if len(cand_str) == len(chal_str) and chal_str in doubled:
                log_structured(logger, 'info', event='pb_challenge_match',
                               iteration=iteration,
                               challenge_name=name,
                               challenge=chal_str,
                               candidate=cand_str,
                               score=float(score))
                if matches_path is not None:
                    record = {
                        'iteration': iteration,
                        'timestamp': datetime.now().isoformat(),
                        'challenge_name': name,
                        'challenge_word': chal_str,
                        'candidate_word': cand_str,
                        'factor_word': json.loads(to_json(fw, rm)),
                        'score': float(score),
                        'scorer_type': scorer_type,
                    }
                    with Path(matches_path).open('a') as mf:
                        mf.write(json.dumps(record) + '\n')
                terminate = True

    if cand_lengths:
        length_stats = {
            'count': len(cand_lengths),
            'min': min(cand_lengths),
            'max': max(cand_lengths),
            'mean': statistics.mean(cand_lengths),
            'median': statistics.median(cand_lengths),
            'avg_goal': statistics.mean([len(chal_str) for chal_str in challenge_strs.values()]),
        }
        log_structured(logger, 'info', event='pb_candidate_length_stats',
                       iteration=iteration, **length_stats)

    if cand_red_lengths:
        red_length_stats = {
            'count': len(cand_red_lengths),
            'min': min(cand_red_lengths),
            'max': max(cand_red_lengths),
            'mean': statistics.mean(cand_red_lengths),
            'median': statistics.median(cand_red_lengths),
            'avg_goal': statistics.mean([len(chal_str) for chal_str in challenge_strs.values()]),
        }
        log_structured(logger, 'info', event='pb_candidate_2.5_reduced_length_stats',
                       iteration=iteration, **red_length_stats)

    return terminate


# ---------- main loop ----------

def run_pattern_boost(cfg: WorkflowConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = cfg.logging.log_file or str(out_dir / 'run.jsonl')
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=log_file, level=cfg.logging.level)
    logger = logging.getLogger('pattern_boost.workflow')
    matches_path = log_path.parent / f'{log_path.stem}.matches.jsonl'

    log_structured(logger, 'info', event='pb_start', config=asdict(cfg))

    rm = RelatorMap(build_standard_relators(cfg.relators.max_relator_len, cfg.relators.num_generators))
    log_structured(logger, 'info', event='pb_relator_map_built', size=len(rm))

    if not cfg.challenges.path:
        raise ValueError('challenges.path must be set in config')
    total_in_file = len(json.loads(Path(cfg.challenges.path).read_text()))
    challenge_words, challenge_strs = load_challenges(
        cfg.challenges.path,
        sample_size=cfg.challenges.sample_size,
        seed=cfg.seed,
    )
    log_structured(logger, 'info', event='pb_challenges_loaded',
                   count=len(challenge_words), available=total_in_file,
                   requested=cfg.challenges.sample_size,
                   path=cfg.challenges.path,
                   challenge_names=list(challenge_strs.keys()))

    scorer = make_scorer(cfg.scoring.type, challenge_words, cfg.relators.num_generators)
    ls_cfg = _make_local_search_cfg(cfg.local_search, cfg.relators.num_generators)
    searcher = LocalSearcher(rm, scorer, ls_cfg)

    sample_cfg = _make_sample_cfg(cfg.initial_sampling, cfg.relators.num_generators)
    pool = list(generate_words(rm, sample_cfg, cfg.initial_sampling.n_initial, seed=cfg.seed))
    log_structured(logger, 'info', event='pb_initial_pool', size=len(pool))

    initial_scores = [scorer.score_sample(fw, rm) for fw in pool]
    if len(pool) > cfg.workflow.max_pool_size:
        scored = sorted(zip(initial_scores, pool), key=lambda x: -x[0])[:cfg.workflow.max_pool_size]
        initial_scores = [s for s, _ in scored]
        pool = [fw for _, fw in scored]
        log_structured(logger, 'info', event='pb_initial_pool_truncated',
                       kept=len(pool), cap=cfg.workflow.max_pool_size)

        if check_stopping_criteria(scored, rm, challenge_strs, logger, 0):
            log_structured(logger, 'info', event='pb_stopped_early',
                           iteration=0, reason='challenge_match')
            return


    tokenizer = FactoredWordTokenizer(num_generators=cfg.relators.num_generators, num_relators=len(rm))
    if cfg.tokenizer.use_bpe:
        tokenizer.train_bpe(pool, vocab_size=cfg.tokenizer.vocab_size)
    tokenizer.save(out_dir / 'tokenizer')
    log_structured(logger, 'info', event='pb_tokenizer_ready',
                   use_bpe=cfg.tokenizer.use_bpe,
                   vocab_size=tokenizer.vocab_size,
                   base_size=len(tokenizer.base_tokens))

    device = pick_device()
    if cfg.init_model_from:
        model = load_checkpoint(cfg.init_model_from)
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f'checkpoint vocab_size={model.config.vocab_size} does not match '
                f'tokenizer vocab_size={tokenizer.vocab_size} '
                f'(loaded from {cfg.init_model_from})'
            )
        log_structured(logger, 'info', event='pb_model_loaded_from_ckpt',
                       device=str(device), vocab_size=tokenizer.vocab_size,
                       path=cfg.init_model_from)
    else:
        model = build_model(tokenizer, cfg.model)
        log_structured(logger, 'info', event='pb_model_built',
                       device=str(device), vocab_size=tokenizer.vocab_size)

    log_structured(logger, 'info', event='pb_initial_scores', **_score_summary(initial_scores))

    for it in range(cfg.workflow.num_iterations):
        log_structured(logger, 'info', event='pb_iter_start', iteration=it)

        if cfg.workflow.train_from_scratch and it > 0:
            _release_model(model)
            model = build_model(tokenizer, cfg.model)
            log_structured(logger, 'info', event='pb_model_reinitialized',
                           iteration=it, vocab_size=tokenizer.vocab_size)

        model_train(model, tokenizer, pool, cfg.training, device=device, logger=logger)

        gen_cfg = SampleGenConfig(
            n=cfg.generation.n_samples,
            max_length=cfg.generation.max_length,
            temperature=cfg.generation.temperature,
            top_k=cfg.generation.top_k,
            top_p=cfg.generation.top_p,
            seed=cfg.seed + it,
            batch_size=cfg.generation.batch_size,
        )
        candidates = model_sample(model, tokenizer, gen_cfg, device=device)
        log_structured(logger, 'info', event='pb_sampled',
                       iteration=it, valid=len(candidates),
                       requested=cfg.generation.n_samples)

        if not candidates:
            log_structured(logger, 'warning', event='pb_no_valid_samples', iteration=it)
            continue

        results: list[tuple[float, object]] = []
        workers = cfg.workflow.parallel_workers
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(searcher.search, fw, cfg.seed + it * 1000 + i): i
                for i, fw in enumerate(candidates)
            }
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f'iter {it} search'):
                score, best_fw = fut.result()
                results.append((score, best_fw))

        results.sort(key=lambda x: -x[0])
        post_scores = [s for s, _ in results]
        log_structured(logger, 'info', event='pb_search_scores',
                       iteration=it, **_score_summary(post_scores))

        # Log top 3 highest-scored expanded words (results are sorted by score descending)
        top_3_words = []
        for score, fw in results[:3]:
            expanded_str = word_to_str(rm.expand(fw.factors))
            top_3_words.append({'score': float(score), 'word': expanded_str})

        if top_3_words:
            log_structured(logger, 'info', event='pb_top_3_candidates',
                           iteration=it, candidates=top_3_words)

        if check_stopping_criteria(results, rm, challenge_strs, logger, it,
                                   matches_path=matches_path, scorer_type=cfg.scoring.type):
            log_structured(logger, 'info', event='pb_stopped_early',
                           iteration=it, reason='challenge_match')
            break

        top = [fw for _, fw in results[:cfg.workflow.top_k]]
        pool = _merge_pool(pool, top, rm, cfg.workflow.max_pool_size)
        log_structured(logger, 'info', event='pb_pool_updated',
                       iteration=it, pool_size=len(pool), kept=len(top))

        save_iter = (it % cfg.workflow.save_every == 0) or (it == cfg.workflow.num_iterations - 1)
        if save_iter:
            iter_dir = out_dir / f'iter_{it:04d}'
            iter_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, iter_dir / 'model')
            (iter_dir / 'best_words.json').write_text(
                json.dumps(_serialize_pool(top, rm), indent=2)
            )
            (iter_dir / 'scores.json').write_text(
                json.dumps([float(s) for s in post_scores], indent=2)
            )
            log_structured(logger, 'info', event='pb_iter_saved',
                           iteration=it, path=str(iter_dir))

    log_structured(logger, 'info', event='pb_done',
                   total_iterations=cfg.workflow.num_iterations,
                   final_pool_size=len(pool))
