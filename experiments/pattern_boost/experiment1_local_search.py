"""
Experiment 1: PatternBoost local search seeded from standard power relators,
scored against shortlex challenge relators.

Created by: Alex Myasnikov
Date: April 24, 2026

Pipeline:
  1. Load shortlex relators from test_relators.json → challenge Words
  2. Build RelatorMap from standard W^5 power relators (|W| <= MAX_RELATOR_LEN)
  3. Generate N random trivial words, score each against challenges
  4. Run local search on each, score again
  5. Report per-sample improvement and check for exact challenge matches
"""

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from algorithms.word_problem import word_from_str, word_to_str
from algorithms.pattern_boost.pb_tcgraph import (
    RelatorMap,
    SampleConfig,
    LocalSearchConfig,
    LocalSearcher,
    HellingerScorer,
    HammingScorer,
    EditScorer,
    generate_words,
    uniform_len,
    build_standard_relators,
    edit_distance,
    hamming_distance,
)

# Todo: move to config file
# RELATORS_FILE = Path(__file__).parent.parent.parent / "test_relators.json"
RELATORS_FILE = Path(__file__).parent / "test_relators_1.json"
NUM_SAMPLES = 100
SEED = 42
MAX_RELATOR_LEN = 5
NUM_GENERATORS = 2


def load_challenges(path: Path) -> tuple[list, dict[str, str]]:
    """Load shortlex relators as challenge Word list + name→word_str map."""
    data = json.loads(path.read_text())
    relator_list = [(name, entry["word"]) for name, entry in data.items()]
    challenge_words = [word_from_str(w) for _, w in relator_list]
    challenge_strs = {name: w for name, w in relator_list}
    return challenge_words, challenge_strs


def experiment(cfg, ls_cfg, score_type):
    SCORER_TYPE = score_type

    # 1. Load challenges (shortlex relators)
    print("Loading shortlex challenge relators...")
    challenge_words, challenge_strs = load_challenges(RELATORS_FILE)
    print(f"  {len(challenge_words)} challenges loaded")

    if SCORER_TYPE == "edit":
        scorer = EditScorer(challenge_words)
    elif SCORER_TYPE == "hellinger":
        scorer = HellingerScorer(challenge_words, NUM_GENERATORS)
    elif SCORER_TYPE == "hamming":
        scorer = HammingScorer(challenge_words)
    else:
        raise ValueError(f"Unknown scorer type: {SCORER_TYPE}")

    # 2. Build RelatorMap from standard W^5 power relators
    print(f"\nBuilding standard relator map (|W| <= {MAX_RELATOR_LEN})...")
    std_relators = build_standard_relators(MAX_RELATOR_LEN)
    rm = RelatorMap(std_relators)
    print(f"  RelatorMap size: {len(rm)}")

    # 3. Generate N random trivial words; score against challenges
    print(f"\nGenerating {NUM_SAMPLES} random trivial words...")
    samples = generate_words(rm, cfg, NUM_SAMPLES, seed=SEED)

    before_scores = [scorer.score_sample(fw, rm) for fw in samples]
    print(f"  Before local search — "
          f"min={min(before_scores):.4f}  "
          f"max={max(before_scores):.4f}  "
          f"mean={sum(before_scores) / len(before_scores):.4f}")

    # 4. Run local search; score after
    print(f"\nRunning local search (max_steps={ls_cfg.max_steps}, beam_width={ls_cfg.beam_width})...")
    searcher = LocalSearcher(rm, HellingerScorer(challenge_words, NUM_GENERATORS), ls_cfg)

    def search_worker(args):
        fw, seed = args
        best_score, best_fw = searcher.search(fw, seed=seed)
        return (best_score, best_fw)

    results: list[tuple[float, object]] = []
    search_args = [(fw, SEED + i) for i, fw in enumerate(samples)]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(search_worker, args): i for i, args in enumerate(search_args)}
        for future in tqdm(as_completed(futures), total=len(samples)):
            results.append(future.result())

    after_scores = [s for s, _ in results]
    print(f"  After  local search — "
          f"min={min(after_scores):.4f}  "
          f"max={max(after_scores):.4f}  "
          f"mean={sum(after_scores) / len(after_scores):.4f}")

    # # 5. Per-sample table
    # print(f"\n{'#':>3}  {'before':>8}  {'after':>8}  {'gain':>8}  {'len':>6}  {'factors':>7}")
    # print("-" * 52)
    # for i, (b, (a, best_fw)) in enumerate(zip(before_scores, results)):
    #     expanded = rm.expand(best_fw.factors)
    #     print(f"{i:>3}  {b:>8.4f}  {a:>8.4f}  {a-b:>+8.4f}  "
    #           f"{len(expanded):>6}  {len(best_fw.factors):>7}")

    # 5.5. Display word-based distances to challenges
    print(f"\nWord-based distances to challenges...")
    for i, (name, w) in enumerate(challenge_strs.items()):
        min_hd = 99999999
        min_ed = 99999999
        min_len_diff = 99999999
        min_cand = ""
        for score, best_fw in results:
            expanded_word = rm.expand(best_fw.factors)
            hd = hamming_distance(expanded_word, word_from_str(w))
            #            print(f"*******  {w} =?= {word_to_str(expanded_word)} : {hd}")
            ed = edit_distance(expanded_word, word_from_str(w))
            #            len_diff = abs(len(expanded_word) - len(w))
            if hd < min_hd:
                min_hd = hd
                min_cand = expanded_word
            if ed < min_ed:
                min_ed = ed
            # if len_diff < min_len_diff:
            #     min_len_diff = len_diff

        print(
            f"  {name}: MAX  hamming={min_hd}  edit={min_ed} len: chal={len(w)} cand={len(min_cand)} diff={abs(len(w) - len(min_cand))}\n"
            f"    chal: {w}\n    cand: {word_to_str(min_cand)}")

    # 6. Check for exact matches against challenge set
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

    # # 7. Top-3 results
    # top = sorted(results, key=lambda x: -x[0])[:3]
    # print(f"\nTop 3 results:")
    # for rank, (score, best_fw) in enumerate(top, 1):
    #     expanded = rm.expand(best_fw.factors)
    #     expanded_str = word_to_str(expanded)
    #     factor_names = [rm.name(f.relator_idx) for f in best_fw.factors]
    #     print(f"  [{rank}] score={score:.4f}  len={len(expanded)}  "
    #           f"factors={len(best_fw.factors)} {factor_names}")
    #     print(f"       {expanded_str[:80]}")


def main():
    # Configs Todo: move to config file
    cfg = SampleConfig()
    cfg.min_factors = 2
    cfg.max_factors = 10
    cfg.conj_len = uniform_len(4)
    cfg.min_expanded_len = 10

    ls_cfg = LocalSearchConfig()
    ls_cfg.max_steps = 50
    ls_cfg.beam_width = 10
    ls_cfg.num_conj_samples = 5
    ls_cfg.num_insert_samples = 5
    ls_cfg.conj_max_len = 4

    count = 1
    for max_steps in [10, 50, 100]:
        for beam_width in [1, 10, 50]:
            for score_type in ["hamming", "edit"]:
                ls_cfg.beam_width = beam_width
                ls_cfg.max_steps = max_steps

                exp_config = ls_cfg.to_dict()
                exp_config.update({
                    "score_type": score_type,
                    "num_samples": NUM_SAMPLES,
                    "count": count
                })
                print(f"\n\nExp{count} {'=' * 100}")
                print(f"Config: {exp_config}")

                experiment(cfg, ls_cfg, score_type)

                count += 1


if __name__ == "__main__":
    main()
