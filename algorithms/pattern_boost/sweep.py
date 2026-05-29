# Created by: Alex Myasnikov
# Date: May 20, 2026

"""Hyperparameter sweep orchestration for PatternBoost.

Each sweep is a cartesian product over one or more config parameters
(specified as dot-notation paths into WorkflowConfig). For each parameter
combination, the workflow is executed `num_trials` times, each time on a
non-overlapping sample of challenges drawn from the challenge file defined in
the base config.  Runs are sequential; results are aggregated into
sweep_summary.json at the sweep root.

Directory layout:
    <output_dir>/sweep_<timestamp>/
        sweep_summary.json
        combo_0000_<slug>/
            trial_0000/   (each trial is a normal run_pattern_boost output dir)
            trial_0001/
            ...
        combo_0001_<slug>/
            ...
"""

import copy
import itertools
import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from algorithms.logger import log_structured
from algorithms.pattern_boost.workflow import RunSummary, WorkflowConfig, load_config, run_pattern_boost


@dataclass
class SweepConfig:
    base_config: str              # path to base WorkflowConfig YAML
    output_dir: str = 'output/pb_sweep'
    seed: int = 42
    num_trials: int = 1           # runs per combo; each trial uses a non-overlapping challenge sample
    params: dict = field(default_factory=dict)
    # Keys: dot-notation paths into WorkflowConfig (e.g. "local_search.beam_width")
    # Values: list of values to sweep — cartesian product over all keys
    derived: dict = field(default_factory=dict)
    # Derived parameters computed after param overrides are applied.
    # Keys: dot-notation paths to set.
    # Values: Python expressions evaluated with config sub-objects as context.
    # Example:
    #   derived:
    #     workflow.top_k: "workflow.max_pool_size"
    #     generation.n_samples: "int(workflow.max_pool_size * 1.1)"


def load_sweep_config(path) -> SweepConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return SweepConfig(
        base_config=raw['base_config'],
        output_dir=raw.get('output_dir', SweepConfig.output_dir),
        seed=raw.get('seed', SweepConfig.seed),
        num_trials=raw.get('num_trials', SweepConfig.num_trials),
        params=raw.get('params', {}),
        derived=raw.get('derived', {}),
    )


def _apply_override(cfg: WorkflowConfig, dotted_key: str, value) -> None:
    """Set a nested config field using a dot-notation path (mutates cfg in-place)."""
    parts = dotted_key.split('.')
    obj = cfg
    for part in parts[:-1]:
        if not hasattr(obj, part):
            raise AttributeError(
                f"Config path {dotted_key!r}: no attribute {part!r} on {type(obj).__name__}"
            )
        obj = getattr(obj, part)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise AttributeError(
            f"Config path {dotted_key!r}: no attribute {leaf!r} on {type(obj).__name__}"
        )
    setattr(obj, leaf, value)


_DERIVED_BUILTINS = {'int': int, 'float': float, 'round': round, 'max': max, 'min': min}


def _apply_derived(cfg: WorkflowConfig, derived: dict) -> dict:
    """Evaluate and apply derived parameter expressions.

    Each expression is evaluated with the config's top-level sub-objects
    (workflow, generation, local_search, ...) available by name, so natural
    Python attribute access like ``workflow.max_pool_size`` works.

    Returns a dict of {dotted_key: computed_value} for logging.
    """
    ctx = {
        field_name: getattr(cfg, field_name)
        for field_name in cfg.__dataclass_fields__
    }
    applied = {}
    for dotted_key, expr in derived.items():
        value = eval(str(expr), {'__builtins__': _DERIVED_BUILTINS}, ctx)  # noqa: S307
        _apply_override(cfg, dotted_key, value)
        applied[dotted_key] = value
    return applied


def _param_slug(combo: dict) -> str:
    """Human-readable run suffix from a param combo dict."""
    parts = []
    for k, v in combo.items():
        short_k = k.split('.')[-1][:6]
        short_v = str(v).replace(' ', '')[:8]
        parts.append(f'{short_k}{short_v}')
    slug = '_'.join(parts)
    return slug[:40].rstrip('_')


def _score_stats(scores: list[float]) -> dict:
    if not scores:
        return {}
    sorted_s = sorted(scores)
    n = len(sorted_s)
    return {
        'min': sorted_s[0],
        'max': sorted_s[-1],
        'mean': statistics.mean(sorted_s),
        'p50': statistics.median(sorted_s),
        'p90': sorted_s[int(n * 0.9)],
    }


def _partition_challenges(all_names: list[str], sample_size: int, num_trials: int, seed: int) -> list[list[str]]:
    """Shuffle all challenge names and partition into non-overlapping windows.

    Returns a list of `num_trials` windows, each of length `sample_size`.
    If `num_trials * sample_size > len(all_names)`, caps at as many full
    windows as the list allows and logs a warning.
    """
    import random as _random
    shuffled = list(all_names)
    _random.Random(seed).shuffle(shuffled)
    max_trials = len(shuffled) // sample_size
    if num_trials > max_trials:
        logging.getLogger('pattern_boost.sweep').warning(
            'num_trials=%d exceeds available non-overlapping windows=%d '
            '(total_challenges=%d, sample_size=%d); capping at %d',
            num_trials, max_trials, len(shuffled), sample_size, max_trials,
        )
        num_trials = max_trials
    return [shuffled[t * sample_size:(t + 1) * sample_size] for t in range(num_trials)]


def _build_summary(sweep_cfg: SweepConfig, combo_records: list[dict]) -> dict:
    total_runs = sum(len(c['trials']) for c in combo_records)
    total_matched = sum(
        sum(1 for t in c['trials'] if t['matched'])
        for c in combo_records
    )
    all_scores = [t['best_score'] for c in combo_records for t in c['trials']]

    return {
        'sweep_config': asdict(sweep_cfg),
        'total_runs': total_runs,
        'matched_runs': total_matched,
        'match_rate': total_matched / total_runs if total_runs else 0.0,
        'score_stats_all_runs': _score_stats(all_scores),
        'combos': combo_records,
    }


def run_sweep(sweep_cfg: SweepConfig) -> None:
    sweep_root = Path(sweep_cfg.output_dir)
    sweep_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('pattern_boost.sweep')
    log_structured(logger, 'info', event='pb_sweep_start',
                   base_config=sweep_cfg.base_config,
                   output_dir=str(sweep_root),
                   num_trials=sweep_cfg.num_trials,
                   params=sweep_cfg.params)

    base_cfg = load_config(sweep_cfg.base_config)

    # --- pre-compute per-trial challenge partitions ---
    challenge_names_all: list[str] = list(json.loads(Path(base_cfg.challenges.path).read_text()).keys())
    sample_size = base_cfg.challenges.sample_size or len(challenge_names_all)
    use_filter = sweep_cfg.num_trials > 1 or base_cfg.challenges.sample_size is not None

    if sweep_cfg.num_trials > 1:
        trial_windows = _partition_challenges(
            challenge_names_all, sample_size, sweep_cfg.num_trials, sweep_cfg.seed
        )
    else:
        # single trial: respect existing sample_size / names_filter from base config
        trial_windows = [None]  # None means "use base cfg as-is"

    effective_trials = len(trial_windows)
    log_structured(logger, 'info', event='pb_sweep_challenge_partition',
                   total_challenges=len(challenge_names_all),
                   sample_size=sample_size,
                   effective_trials=effective_trials)

    # --- cartesian product of hyperparams ---
    param_keys = list(sweep_cfg.params.keys())
    param_value_lists = [sweep_cfg.params[k] for k in param_keys]
    combos = list(itertools.product(*param_value_lists))

    log_structured(logger, 'info', event='pb_sweep_combos',
                   num_combos=len(combos), num_trials=effective_trials,
                   total_runs=len(combos) * effective_trials,
                   param_keys=param_keys)

    combo_records: list[dict] = []

    for combo_idx, combo_values in enumerate(combos):
        combo = dict(zip(param_keys, combo_values))
        slug = _param_slug(combo)
        combo_dir_name = f'combo_{combo_idx:04d}_{slug}' if slug else f'combo_{combo_idx:04d}'
        combo_dir = sweep_root / combo_dir_name

        log_structured(logger, 'info', event='pb_sweep_combo_start',
                       combo_idx=combo_idx, total_combos=len(combos),
                       params=combo, combo_dir=str(combo_dir))

        trial_records: list[dict] = []

        for trial_idx, trial_names in enumerate(trial_windows):
            trial_dir = combo_dir / f'trial_{trial_idx:04d}'

            cfg = copy.deepcopy(base_cfg)
            for k, v in combo.items():
                _apply_override(cfg, k, v)
            # derived params evaluated after explicit overrides
            derived_values = _apply_derived(cfg, sweep_cfg.derived) if sweep_cfg.derived else {}
            cfg.output_dir = str(trial_dir)
            # unique seed per (combo, trial) so initial pool sampling differs
            cfg.seed = sweep_cfg.seed + combo_idx * 10000 + trial_idx

            if trial_names is not None:
                cfg.challenges.names_filter = trial_names
                cfg.challenges.sample_size = None  # names_filter takes precedence

            log_structured(logger, 'info', event='pb_sweep_trial_start',
                           combo_idx=combo_idx, trial_idx=trial_idx,
                           challenges=trial_names or 'base_cfg',
                           derived=derived_values or None)

            summary: RunSummary = run_pattern_boost(cfg)

            record = {
                'trial_idx': trial_idx,
                'challenge_names': trial_names,
                'matched': summary.matched,
                'match_iteration': summary.match_iteration,
                'match_challenge': summary.match_challenge,
                'best_score': summary.best_score,
                'final_score_stats': summary.final_score_stats,
                'total_iterations': summary.total_iterations,
                'trial_dir': str(trial_dir.relative_to(sweep_root)),
            }
            trial_records.append(record)

            log_structured(logger, 'info', event='pb_sweep_trial_done',
                           combo_idx=combo_idx, trial_idx=trial_idx,
                           matched=summary.matched, best_score=summary.best_score)

        # per-combo aggregate stats
        matched_trials = [t for t in trial_records if t['matched']]
        match_iters = [t['match_iteration'] for t in matched_trials if t['match_iteration'] is not None]
        best_scores = [t['best_score'] for t in trial_records]

        combo_record = {
            'combo_idx': combo_idx,
            'params': combo,
            'combo_dir': combo_dir_name,
            'trials': trial_records,
            'matched_trials': len(matched_trials),
            'match_rate': len(matched_trials) / len(trial_records) if trial_records else 0.0,
            'avg_iterations_to_match': statistics.mean(match_iters) if match_iters else None,
            'score_stats': _score_stats(best_scores),
        }
        combo_records.append(combo_record)

        log_structured(logger, 'info', event='pb_sweep_combo_done',
                       combo_idx=combo_idx, params=combo,
                       matched_trials=combo_record['matched_trials'],
                       match_rate=combo_record['match_rate'])

    summary_data = _build_summary(sweep_cfg, combo_records)
    summary_path = sweep_root / 'sweep_summary.json'
    summary_path.write_text(json.dumps(summary_data, indent=2))

    log_structured(logger, 'info', event='pb_sweep_done',
                   total_runs=summary_data['total_runs'],
                   matched_runs=summary_data['matched_runs'],
                   match_rate=summary_data['match_rate'],
                   summary_path=str(summary_path))
