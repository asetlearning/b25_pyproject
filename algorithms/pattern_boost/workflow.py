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
import time
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
from algorithms.word_problem import word_from_str, word_to_str, greedy_reduce2_5, greedy_reduce_cyclic_exhaustive2_5
from algorithms.pattern_boost.common import load_challenges, make_scorer
from data.relators_io import load_relators
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
    ConjMoveStrategy,
    HybridScorer,
    LocalSearchConfig,
    LocalSearcher,
    RelatorMap,
    SampleConfig,
    ScorerConfig,
    WordCompressor,
    build_hybrid_relators,
    build_standard_relators,
    bpe_model_from_compressor,
    from_json,
    generate_words,
    to_json,
    uniform_len,
)
from algorithms.word_compression import BPETokenizerWordCompression
from algorithms.pattern_boost.tokenizer import FactoredWordTokenizer

_pb = _ext.pattern_boost


# ---------- run result ----------

@dataclass
class RunSummary:
    matched: bool = False
    match_iteration: Optional[int] = None
    match_challenge: Optional[str] = None
    total_iterations: int = 0
    best_score: float = 0.0
    final_score_stats: dict = field(default_factory=dict)


@dataclass
class _RunCtx:
    rm: RelatorMap
    challenge_strs: dict
    logger: logging.Logger
    scorer_type: str
    matches_path: Optional[Path] = None
    compressor: Optional[object] = None


# ---------- config schema ----------

@dataclass
class RelatorsCfg:
    max_relator_len: int = 5
    num_generators: int = 2
    standard_generator_tokens: Optional[list[str]] = None  # resolved to indices via compressor
    relators_file: Optional[str] = None  # if set, load relators from JSON file instead of generating


@dataclass
class ChallengesCfg:
    path: str = ''
    sample_size: Optional[int] = None  # if set, load a random subset of this many
    names_filter: Optional[
        list] = None  # if set, load only challenges with these names (overrides sample_size)


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
    conj_move_strategy: str = 'deterministic'  # deterministic | random


@dataclass
class ScoringCfg:
    type: str = 'hellinger'  # hellinger | hamming | edit | word_product_length | abelianized_product_length | high_power_count | length_range | hybrid
    reduce2_5: bool = False  # apply greedy 2.5 reduction before scoring
    penalize_trivial: bool = False  # score 0.0 for words that reduce to the identity
    components: Optional[list] = None  # for hybrid only: [{type, weight, params?}]
    params: Optional[dict] = None  # for length_range / high_power_count top-level


@dataclass
class WorkflowLoopCfg:
    num_iterations: int = 10
    top_k: int = 100
    max_pool_size: int = 5000
    parallel_workers: int = 4
    save_every: int = -1  # iterations between checkpoints; <= 0 disables all saves
    train_from_scratch: bool = False  # rebuild model with fresh weights each iter


@dataclass
class LoggingCfg:
    level: str = 'info'
    log_file: Optional[str] = None


@dataclass
class CompressionCfg:
    use_compression: bool = False
    compression_tokenizer: Optional[str] = None  # path to BPETokenizerWordCompression model file


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
    compression: CompressionCfg = field(default_factory=CompressionCfg)


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

def _resolve_standard_generators(tokens: list[str], compressor) -> list[int]:
    indices = []
    for tok in tokens:
        word = compressor.encode_str(tok)
        letters = word.get_letters()
        if len(letters) != 1:
            raise ValueError(f"Token '{tok}' encodes to {len(letters)} generators, expected 1")
        indices.append(abs(letters[0]))
    return indices


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
    ls.conj_move_strategy = (ConjMoveStrategy.kRandom if cfg.conj_move_strategy == 'random'
                              else ConjMoveStrategy.kDeterministic)
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


def _is_trivial_abelianization(word: _ext.Word) -> bool:
    """Check if a word is trivial in its abelianization.

    A word is trivial in its abelianization if all its powers are trivial.
    """
    return all(power % 5 == 0 for power in word.count_powers().values())


# ---------- stopping criteria ----------

def check_stopping_criteria(
    ctx: _RunCtx,
    results: list,
    iteration: int,
    perfect_score: float = 1.0,
    tol: float = 1e-6,
) -> tuple[bool, Optional[str]]:
    """Apply per-iteration stopping criteria after local search.

    Criteria:
      1. Log any candidate whose score is (numerically) `perfect_score` —
         informational, does NOT terminate.
      2. Log any candidate whose expanded B(2,5) word is cyclically equal to
         any challenge string. **Terminates** the iteration loop.

    Returns (should_stop, first_matched_challenge_name).
    """
    terminate = False
    first_match: Optional[str] = None

    for score, fw in results:
        if abs(float(score) - perfect_score) < tol:
            expanded_word = ctx.rm.expand(fw.factors)
            if ctx.compressor:
                decoded_str = word_to_str(ctx.compressor.decode_word(expanded_word))
                log_structured(ctx.logger, 'info', event='pb_perfect_score_candidate',
                               iteration=iteration, score=float(score),
                               word=decoded_str,
                               compressed_word=ctx.compressor.compressed_repr(expanded_word))
            else:
                log_structured(ctx.logger, 'info', event='pb_perfect_score_candidate',
                               iteration=iteration, score=float(score),
                               word=word_to_str(expanded_word))

    cand_lengths = []
    cand_red_lengths = []
    nonzero_red_count = 0
    top3_reduced: list[dict] = []
    for score, fw in results:
        expanded_word = ctx.rm.expand(fw.factors)
        if ctx.compressor:
            cand_str = word_to_str(ctx.compressor.decode_word(expanded_word))
            cand_compressed = ctx.compressor.compressed_repr(expanded_word)
        else:
            cand_str = word_to_str(expanded_word)
            cand_compressed = None

        if not cand_str:
            continue

        abelian_check_word = word_from_str(cand_str) if ctx.compressor else expanded_word
        if not _is_trivial_abelianization(abelian_check_word):
            log_structured(ctx.logger, 'error', event='pb_non_trivial_expansion',
                           iteration=iteration,
                           word=cand_str,
                           )
            continue

#        cand_red = greedy_reduce2_5(cand_str)
        cand_red = greedy_reduce_cyclic_exhaustive2_5(cand_str)
        cand_lengths.append(len(cand_str))
        cand_red_lengths.append(len(cand_red))
        if len(cand_red) > 0:
            nonzero_red_count += 1
        if len(top3_reduced) < 3:
            top3_reduced.append({'score': float(score), 'word': cand_red, 'len': len(cand_red)})

        doubled = cand_str + cand_str
        for name, chal_str in ctx.challenge_strs.items():
            if len(cand_str) == len(chal_str) and chal_str in doubled:
                extra = {'compressed_word': cand_compressed} if cand_compressed is not None else {}
                log_structured(ctx.logger, 'info', event='pb_challenge_match',
                               iteration=iteration,
                               challenge_name=name,
                               challenge=chal_str,
                               candidate=cand_str,
                               score=float(score),
                               **extra)
                if ctx.matches_path is not None:
                    record = {
                        'iteration': iteration,
                        'timestamp': datetime.now().isoformat(),
                        'challenge_name': name,
                        'challenge_word': chal_str,
                        'candidate_word': cand_str,
                        'factor_word': json.loads(to_json(fw, ctx.rm)),
                        'score': float(score),
                        'scorer_type': ctx.scorer_type,
                    }
                    if cand_compressed is not None:
                        record['compressed_word'] = cand_compressed
                    with Path(ctx.matches_path).open('a') as mf:
                        mf.write(json.dumps(record) + '\n')
                if first_match is None:
                    first_match = name
                terminate = True

    if cand_lengths:
        length_stats = {
            'count': len(cand_lengths),
            'min': min(cand_lengths),
            'max': max(cand_lengths),
            'mean': statistics.mean(cand_lengths),
            'median': statistics.median(cand_lengths),
            'avg_goal': statistics.mean([len(chal_str) for chal_str in ctx.challenge_strs.values()]),
        }
        log_structured(ctx.logger, 'info', event='pb_candidate_length_stats',
                       iteration=iteration, **length_stats)

    if cand_red_lengths:
        red_length_stats = {
            'count': len(cand_red_lengths),
            'min': min(cand_red_lengths),
            'max': max(cand_red_lengths),
            'mean': statistics.mean(cand_red_lengths),
            'median': statistics.median(cand_red_lengths),
            'avg_goal': statistics.mean([len(chal_str) for chal_str in ctx.challenge_strs.values()]),
        }
        log_structured(ctx.logger, 'info', event='pb_candidate_2.5_reduced_length_stats',
                       iteration=iteration, **red_length_stats,
                       nonzero_reduced_count=nonzero_red_count,
                       top3_reduced_words=top3_reduced)

    return terminate, first_match


# ---------- main loop ----------

def run_pattern_boost(cfg: WorkflowConfig) -> RunSummary:
    # ====== PHASE 1: Setup and initialization ======
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = cfg.logging.log_file or str(out_dir / 'run.jsonl')
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=log_file, level=cfg.logging.level)
    logger = logging.getLogger('pattern_boost.workflow')
    log_structured(logger, 'info', event='pb_start', config=asdict(cfg))
    matches_path = log_path.parent / f'{log_path.stem}.matches.jsonl'

    # ====== PHASE 2: Resolve num_generators (and optionally init compressor) ======
    compressor = None
    num_generators = cfg.relators.num_generators
    if cfg.compression.use_compression:
        _bpe = BPETokenizerWordCompression.from_file(cfg.compression.compression_tokenizer)
        compressor = WordCompressor(bpe_model_from_compressor(_bpe))
#        print(compressor.print_mappings())
        log_structured(logger, 'info', event='pb_compression_enabled',
                       bpe_path=cfg.compression.compression_tokenizer,
                       num_generators=compressor.num_generators,
                       overrides_relators_num_generators=num_generators)
        num_generators = compressor.num_generators

    # ====== PHASE 2: Build relator map and load challenges ======
    if cfg.relators.relators_file is not None:
        raw = load_relators(cfg.relators.relators_file)
        relators = {_ext.Word(entry['word']) for entry in raw.values()}
        relators |= {-w for w in relators}
        named_relators = [(f'R{i}', w) for i, w in enumerate(relators, 1)]
        rm = RelatorMap(named_relators)
        log_structured(logger, 'info', event='pb_relator_map_built', size=len(rm),
                       source='file', path=cfg.relators.relators_file,
                       loaded_from_file=len(raw))
    else:
        std_gens = None
        if cfg.relators.standard_generator_tokens is not None:
            if compressor is None:
                raise ValueError("standard_generator_tokens requires use_compression=true")
            std_gens = _resolve_standard_generators(cfg.relators.standard_generator_tokens, compressor)
        rm = RelatorMap(build_hybrid_relators(cfg.relators.max_relator_len, num_generators, std_gens))
        log_structured(logger, 'info', event='pb_relator_map_built', size=len(rm), source='generated')

    if not cfg.challenges.path:
        raise ValueError('challenges.path must be set in config')

    total_in_file = len(json.loads(Path(cfg.challenges.path).read_text()))
    challenge_words, challenge_strs = load_challenges(
        cfg.challenges.path,
        sample_size=cfg.challenges.sample_size,
        seed=cfg.seed,
        names_filter=cfg.challenges.names_filter,
    )
    log_structured(logger, 'info', event='pb_challenges_loaded',
                   count=len(challenge_words), available=total_in_file,
                   requested=cfg.challenges.sample_size,
                   path=cfg.challenges.path,
                   challenge_names=list(challenge_strs.keys()))

    if compressor:
        encoded_challenges = {}
        for name, word_str in challenge_strs.items():
            compressed_letters = list(compressor.encode_str(word_str).get_letters())
            encoded_challenges[name] = {
                'word_len': len(word_str),
                'compressed_word': compressed_letters,
                'compressed_word_len': len(compressed_letters),
            }
        log_structured(logger, 'info', event='pb_challenges_encoded',
                       challenges=encoded_challenges)

    scorer_config = None
    scorer_num_generators = num_generators
    if compressor:
        scorer_config = ScorerConfig(
            word_compressor=compressor,
            reduce2_5=cfg.scoring.reduce2_5,
            penalize_trivial=cfg.scoring.penalize_trivial)
        scorer_num_generators = cfg.relators.num_generators
        log_structured(logger, 'info', event='pb_scorer_decode_compressed',
                       scorer_num_generators=scorer_num_generators)
    elif cfg.scoring.reduce2_5 or cfg.scoring.penalize_trivial:
        scorer_config = ScorerConfig(
            reduce2_5=cfg.scoring.reduce2_5,
            penalize_trivial=cfg.scoring.penalize_trivial)

    ctx = _RunCtx(
        rm=rm,
        challenge_strs=challenge_strs,
        logger=logger,
        scorer_type=cfg.scoring.type,
        matches_path=matches_path,
        compressor=compressor,
    )

    # ====== PHASE 3: Initialize scorer and local searcher ======
    scorer = make_scorer(
        cfg.scoring.type, challenge_words, scorer_num_generators,
        scorer_config, components=cfg.scoring.components, params=cfg.scoring.params
    )
    ls_cfg = _make_local_search_cfg(cfg.local_search, num_generators)
    searcher = LocalSearcher(rm, scorer, ls_cfg)

    # ====== PHASE 4: Generate initial pool ======
    sample_cfg = _make_sample_cfg(cfg.initial_sampling, num_generators)
    pool = list(generate_words(rm, sample_cfg, cfg.initial_sampling.n_initial, seed=cfg.seed))
    log_structured(logger, 'info', event='pb_initial_pool', size=len(pool))

    initial_scores = [scorer.score_sample(fw, rm) for fw in pool]
    if len(pool) > cfg.workflow.max_pool_size:
        scored = sorted(zip(initial_scores, pool), key=lambda x: -x[0])[:cfg.workflow.max_pool_size]
        initial_scores = [s for s, _ in scored]
        pool = [fw for _, fw in scored]
        log_structured(logger, 'info', event='pb_initial_pool_truncated',
                       kept=len(pool), cap=cfg.workflow.max_pool_size)

        stop, match_name = check_stopping_criteria(ctx, scored, 0)
        if stop:
            log_structured(logger, 'info', event='pb_stopped_early',
                           iteration=0, reason='challenge_match')
            return RunSummary(
                matched=True, match_iteration=0, match_challenge=match_name,
                total_iterations=0,
                best_score=float(scored[0][0]) if scored else 0.0,
            )

    # ====== PHASE 5: Build tokenizer ======
    tokenizer = FactoredWordTokenizer(num_generators=num_generators, num_relators=len(rm))
    if cfg.tokenizer.use_bpe:
        tokenizer.train_bpe(pool, vocab_size=cfg.tokenizer.vocab_size)
    tokenizer.save(out_dir / 'tokenizer')
    log_structured(logger, 'info', event='pb_tokenizer_ready',
                   use_bpe=cfg.tokenizer.use_bpe,
                   vocab_size=tokenizer.vocab_size,
                   base_size=len(tokenizer.base_tokens))

    # ====== PHASE 6: Initialize or load model ======
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

    summary = RunSummary()
    last_post_scores: list[float] = []

    # ====== PHASE 7: Main iteration loop (train → sample → search → score → keep top) ======
    for it in range(cfg.workflow.num_iterations):
        log_structured(logger, 'info', event='pb_iter_start', iteration=it)

        # Optionally reinitialize model from scratch
        if cfg.workflow.train_from_scratch and it > 0:
            _release_model(model)
            model = build_model(tokenizer, cfg.model)
            log_structured(logger, 'info', event='pb_model_reinitialized',
                           iteration=it, vocab_size=tokenizer.vocab_size)

        # Train model on current pool
        model_train(model, tokenizer, pool, cfg.training, device=device, logger=logger)

        # Sample new candidates from model
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

        # Local search on each candidate in parallel
        results: list[tuple[float, object]] = []
        search_metadata: list[dict] = []
        search_times: list[float] = []
        workers = cfg.workflow.parallel_workers

        def _timed_search(fw, seed):
            t0 = time.perf_counter()
            sr = searcher.search(fw, seed)
            return sr, time.perf_counter() - t0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_timed_search, fw, cfg.seed + it * 1000 + i): i
                for i, fw in enumerate(candidates)
            }
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f'iter {it} search'):
                sr, elapsed = fut.result()
                results.append((sr.score, sr.best_fw))
                search_metadata.append(sr.metadata)
                search_times.append(elapsed)

        if search_metadata:
            nbr_sizes = [m["avg_neighborhood_size"] for m in search_metadata
                         if "avg_neighborhood_size" in m]
            word_lens = [m["avg_expanded_word_len"] for m in search_metadata
                         if "avg_expanded_word_len" in m]
            log_structured(logger, 'info', event='pb_search_metadata',
                           iteration=it,
                           avg_neighborhood_size=statistics.mean(nbr_sizes) if nbr_sizes else None,
                           avg_expanded_word_len=statistics.mean(word_lens) if word_lens else None,
                           avg_worker_time_s=statistics.mean(search_times) if search_times else None)

        # Score and rank results
        results.sort(key=lambda x: -x[0])
        post_scores = [s for s, _ in results]
        last_post_scores = post_scores
        log_structured(logger, 'info', event='pb_search_scores',
                       iteration=it, **_score_summary(post_scores))

        # Log top 3 highest-scored expanded words (results are sorted by score descending)
        top_3_words = []
        for score, fw in results[:3]:
            expanded_word = rm.expand(fw.factors)
            factors = fw.factors
            num_factors = len(factors)
            conj_lens = [len(list(f.conjugator.get_letters())) for f in factors]
            avg_conj_len = statistics.mean(conj_lens) if conj_lens else 0.0
            token_ids = tokenizer.encode(fw)
            factor_str = ' '.join(tokenizer._fw_to_base_tokens(fw))
            if compressor:
                word_str = word_to_str(compressor.decode_word(expanded_word))
                compressed_letters = list(expanded_word.get_letters())
                entry = {
                    'score': float(score),
                    'word': word_str,
                    'word_len': len(word_str),
                    'compressed_word': compressed_letters,
                    'compressed_word_len': len(compressed_letters),
                    'num_factors': num_factors,
                    'avg_conj_len': avg_conj_len,
                    'token_ids': token_ids,
                    'token_len': len(token_ids),
                    'factor_str': factor_str,
                }
            else:
                word_str = word_to_str(expanded_word)
                entry = {
                    'score': float(score),
                    'word': word_str,
                    'word_len': len(word_str),
                    'num_factors': num_factors,
                    'avg_conj_len': avg_conj_len,
                    'token_ids': token_ids,
                    'token_len': len(token_ids),
                    'factor_str': factor_str,
                }
            if isinstance(scorer, HybridScorer) and cfg.scoring.components:
                breakdown = scorer.score_word_components(expanded_word)
                entry['component_scores'] = [
                    {'type': comp['type'], 'weight': float(comp.get('weight', 1.0)), 'score': float(s)}
                    for comp, (s, _w) in zip(cfg.scoring.components, breakdown)
                ]
            top_3_words.append(entry)

        if top_3_words:
            log_structured(logger, 'info', event='pb_top_3_candidates',
                           iteration=it, candidates=top_3_words)

        # Check stopping criteria
        stop, match_name = check_stopping_criteria(ctx, results, it)
        if stop:
            summary.matched = True
            summary.match_iteration = it
            summary.match_challenge = match_name
            summary.total_iterations = it + 1
            summary.best_score = float(post_scores[0]) if post_scores else 0.0
            summary.final_score_stats = _score_summary(post_scores) if post_scores else {}
            log_structured(logger, 'info', event='pb_stopped_early',
                           iteration=it, reason='challenge_match')
            break

        # Keep top-k results and merge into pool
        top = [fw for _, fw in results[:cfg.workflow.top_k]]
        pool = _merge_pool(pool, top, rm, cfg.workflow.max_pool_size)
        log_structured(logger, 'info', event='pb_pool_updated',
                       iteration=it, pool_size=len(pool), kept=len(top))

        # Periodically save checkpoints
        save_iter = cfg.workflow.save_every > 0 and (
            (it % cfg.workflow.save_every == 0) or (it == cfg.workflow.num_iterations - 1)
        )
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

    # ====== PHASE 8: Finalize and cleanup ======
    if not summary.matched and last_post_scores:
        summary.total_iterations = cfg.workflow.num_iterations
        summary.best_score = float(last_post_scores[0])
        summary.final_score_stats = _score_summary(last_post_scores)

    log_structured(logger, 'info', event='pb_done',
                   total_iterations=summary.total_iterations,
                   final_pool_size=len(pool))
    _release_model(model)
    return summary
