"""
Experiment 1: PatternBoost local search seeded from standard power relators,
scored against shortlex challenge relators.

Created by: Alex Myasnikov
Date: April 24, 2026

Pipeline:
  1. Load shortlex relators from relators_shortlex_rules_2.5reduced.json → challenge Words
  2. Build RelatorMap from standard W^5 power relators (|W| <= max_relator_len)
  3. Generate N random trivial words, score each against challenges
  4. Run local search on each, score again
  5. Report per-sample improvement and check for exact challenge matches

Usage:
    python -m experiments.pattern_boost.experiment1_local_search <config.yaml>

See configs/experiment1_local_search.yaml for the schema.
"""

from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm
import typer

from algorithms.word_problem import word_from_str, word_to_str
from algorithms.pattern_boost.common import load_challenges, make_scorer
from algorithms.pattern_boost.pb_tcgraph import (
    LocalSearchConfig,
    LocalSearcher,
    RelatorMap,
    SampleConfig,
    build_standard_relators,
    edit_distance,
    generate_words,
    hamming_distance,
    to_json,
    uniform_len,
)
from data.relators_io import save_relators


def experiment(cfg: dict, sample_cfg: SampleConfig, ls_cfg: LocalSearchConfig, score_type: str):
    relators_file = Path(cfg["relators_file"])
    output_dir = Path(cfg["output_dir"])
    num_samples = cfg["num_samples"]
    seed = cfg["seed"]
    max_relator_len = cfg["max_relator_len"]
    num_generators = cfg["num_generators"]
    workers = cfg.get("workers", 10)

    print("Loading challenge relators...")
    challenge_words, challenge_strs = load_challenges(relators_file)
    print(f"  {len(challenge_words)} challenges loaded")

    scorer = make_scorer(score_type, challenge_words, num_generators)

    print(f"\nBuilding standard relator map (|W| <= {max_relator_len})...")
    std_relators = build_standard_relators(max_relator_len)
    rm = RelatorMap(std_relators)
    print(f"  RelatorMap size: {len(rm)}")

    print(f"\nGenerating {num_samples} random trivial words...")
    samples = generate_words(rm, sample_cfg, num_samples, seed=seed)

    before_scores = [scorer.score_sample(fw, rm) for fw in samples]
    print(f"  Before local search — "
          f"min={min(before_scores):.4f}  "
          f"max={max(before_scores):.4f}  "
          f"mean={sum(before_scores) / len(before_scores):.4f}")

    print(f"\nRunning local search (max_steps={ls_cfg.max_steps}, beam_width={ls_cfg.beam_width})...")
    searcher = LocalSearcher(rm, scorer, ls_cfg)

    def search_worker(args):
        fw, s = args
        best_score, best_fw = searcher.search(fw, seed=s)
        return (best_score, best_fw)

    results: list[tuple[float, object]] = []
    search_args = [(fw, seed + i) for i, fw in enumerate(samples)]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(search_worker, a): i for i, a in enumerate(search_args)}
        for future in tqdm(as_completed(futures), total=len(samples)):
            results.append(future.result())

    after_scores = [s for s, _ in results]
    print(f"  After  local search — "
          f"min={min(after_scores):.4f}  "
          f"max={max(after_scores):.4f}  "
          f"mean={sum(after_scores) / len(after_scores):.4f}")

    for score, best_fw in results:
        if abs(score - 1.0) < 1e-6:
            expanded_word = rm.expand(best_fw.factors)
            print(f"  Perfect score (1.0000): {word_to_str(expanded_word)}")

    print(f"\nWord-based distances to challenges...")
    for name, w in challenge_strs.items():
        min_hd = 99999999
        min_ed = 99999999
        min_cand = ""
        for score, best_fw in results:
            expanded_word = rm.expand(best_fw.factors)
            hd = hamming_distance(expanded_word, word_from_str(w))
            ed = edit_distance(expanded_word, word_from_str(w))
            if hd < min_hd:
                min_hd = hd
                min_cand = expanded_word
            if ed < min_ed:
                min_ed = ed
        print(
            f"  {name}: MAX  hamming={min_hd}  edit={min_ed} len: chal={len(w)} "
            f"cand={len(min_cand)} diff={abs(len(w) - len(min_cand))}\n"
            f"    chal: {w}\n    cand: {word_to_str(min_cand)}")

    # Save post-search FactoredWords in pb.to_json format (re-ingestable by
    # train_from_factor_words.py --input). Scores and config saved to a sibling file.
    stem = f"results_{score_type}_b{ls_cfg.beam_width}_s{ls_cfg.max_steps}_n{num_samples}"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.json"
    entries = [json.loads(to_json(best_fw, rm)) for _, best_fw in results]
    out_path.write_text(json.dumps(entries, indent=2))
    print(f"\nSaved {len(entries)} post-search FactoredWords to {out_path}")

    scores_path = output_dir / f"{stem}.scores.json"
    scores_path.write_text(json.dumps([float(s) for s, _ in results], indent=2))
    print(f"Saved scores to {scores_path}")

    config_path = output_dir / f"{stem}.config.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    print(f"Saved configuration to {config_path}")

    print(f"\nChecking for exact challenge matches in best words...")
    matches = 0
    for score, best_fw in results:
        expanded_str = word_to_str(rm.expand(best_fw.factors))
        if expanded_str in challenge_strs.values():
            name = next(n for n, w in challenge_strs.items() if w == expanded_str)
            print(f"  MATCH {name}! score={score:.4f}  word={expanded_str[:70]}")
            matches += 1
    if matches == 0:
        print("  No exact matches.")


def build_sample_cfg(cfg: dict) -> SampleConfig:
    s = cfg["sampling"]
    sc = SampleConfig()
    sc.min_factors = s["min_factors"]
    sc.max_factors = s["max_factors"]
    sc.conj_len = uniform_len(s["conj_len_max"])
    sc.min_expanded_len = s["min_expanded_len"]
    sc.num_generators = cfg["num_generators"]
    return sc


def build_ls_cfg(cfg: dict, max_steps: int, beam_width: int) -> LocalSearchConfig:
    ls = cfg["local_search"]
    lsc = LocalSearchConfig()
    lsc.max_steps = max_steps
    lsc.beam_width = beam_width
    lsc.num_conj_samples = ls["num_conj_samples"]
    lsc.num_insert_samples = ls["num_insert_samples"]
    lsc.conj_max_len = ls["conj_max_len"]
    return lsc


def sweep_experiment(cfg: dict):
    sample_cfg = build_sample_cfg(cfg)
    sweep = cfg["sweep"]
    count = 1
    for max_steps in sweep["max_steps"]:
        for beam_width in sweep["beam_width"]:
            for score_type in sweep["score_types"]:
                ls_cfg = build_ls_cfg(cfg, max_steps, beam_width)
                exp_config = ls_cfg.to_dict()
                exp_config.update({
                    "score_type": score_type,
                    "num_samples": cfg["num_samples"],
                    "count": count,
                })
                print(f"\n\nExp{count} {'=' * 100}")
                print(f"Config: {exp_config}")
                experiment(cfg, sample_cfg, ls_cfg, score_type)
                count += 1


def generate_samples(cfg: dict, num_samples: int | None, seed: int | None):
    sample_cfg = build_sample_cfg(cfg)

    output_dir = Path(cfg["output_dir"])
    if num_samples is None:
        num_samples = cfg["num_samples"]

    if seed is None:
        seed = cfg["seed"]
    else:
        cfg["seed"] = seed

    max_relator_len = cfg["max_relator_len"]

    print(f"\nBuilding standard relator map (|W| <= {max_relator_len})...")
    std_relators = build_standard_relators(max_relator_len)
    rm = RelatorMap(std_relators)
    print(f"  RelatorMap size: {len(rm)}")

    print(f"\nGenerating {num_samples} random trivial words...")
    sample = generate_words(rm, sample_cfg, num_samples, seed=seed)


    # Compute length statistics
    expanded_words = [rm.expand(fw.factors) for fw in sample]
    expanded_lengths = [len(w) for w in expanded_words]
    min_len = min(expanded_lengths)
    max_len = max(expanded_lengths)
    mean_len = sum(expanded_lengths) / len(expanded_lengths)
    sorted_lengths = sorted(expanded_lengths)
    median_len = sorted_lengths[len(sorted_lengths) // 2]

    print(f"  Expanded word lengths — "
          f"min={min_len}  "
          f"max={max_len}  "
          f"mean={mean_len:.2f}  "
          f"median={median_len}")

    # Save the sample
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"relators_sample_n{num_samples}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = output_dir / f"{stem}.factorwords.json"
    entries = [json.loads(to_json(fw, rm)) for fw in sample]
    out_path.write_text(json.dumps(entries))
    print(f"\nSaved {len(entries)} generated samples to {out_path}")

    # Save expanded words in relators format
    expanded_path = output_dir / f"{stem}.json"
    expanded_dict = {f'R{i}': {"word": w.get_letters(), "version": 1} for i, w in enumerate(expanded_words)}
    save_relators(str(expanded_path), expanded_dict)
    print(f"Saved {len(expanded_dict)} expanded words to {expanded_path}")

    config_path = output_dir / f"{stem}.config.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    print(f"Saved configuration to {config_path}")

app = typer.Typer(help="Local search experiment: Config-driven generates and improves trivial words "
                       "based on challenge relators similarity.")


@app.command()
def sweep(
    config: Path =typer.Argument(..., help="Path to experiment YAML config")
):
    """Run experiment sweep over beam steps, width and score functions."""
    cfg = yaml.safe_load(config.read_text())
    print("Configuration:")
    print(json.dumps(cfg, indent=2))

    sweep_experiment(cfg)


@app.command()
def sample(
    config: Path =typer.Argument(..., help="Path to experiment YAML config"),
    num_samples: Optional[int] = typer.Option(None, "--n", help="Number of samples to generate"),
    seed: Optional[int] = typer.Option(None, "--s", help="Seed for random number generator"),
):
    """Run experiment sweep over beam steps, width and score functions."""
    cfg = yaml.safe_load(config.read_text())
    print("Configuration:")
    print(json.dumps(cfg, indent=2))

    generate_samples(cfg, num_samples, seed)

if __name__ == "__main__":
    app()
