# PatternBoost Experiments

PatternBoost is a local-search framework for generating trivial words in B(2,5) that closely resemble a set of **challenge words** in bigram-frequency space. The approach has three phases:

1. **Sample** — generate trivial words as products of conjugated relators
2. **Score** — measure bigram-frequency similarity to challenge words (Hellinger distance)
3. **Search** — run beam-search local search to improve the score

## Architecture

### C++ library (`cpp/tcgraph/pattern_boost/`)

| File | Contents |
|------|----------|
| `pb_relator_map.h/cpp` | `RelatorMap`, `Factor`, `buildStandardRelators` |
| `pb_sample_generator.h/cpp` | `FactoredWord`, `SampleConfig`, `generateWord/s`, JSON I/O |
| `pb_scoring.h/cpp` | `WordScorer` (abstract), `HellingerScorer`, `HammingScorer`, `EditScorer`, `WordProductLengthScorer`, `AbelianizedProductLengthScorer`, `HighPowerCountScorer`, `LengthRangeScorer`, `HybridScorer`, `BigramFrequencies`, `loadChallenges` |
| `pb_local_search.h/cpp` | `LocalSearchConfig`, `LocalSearcher` |

### Python bindings (`tcgraph_ext.pattern_boost`)

Exposed as a submodule of `tcgraph_ext` (same `.so` as the rest of the C++ interface). Built in `cpp/bindings_pattern_boost.cpp`, registered via `init_pattern_boost()` called from `cpp/bindings.cpp`.

### Python wrapper (`algorithms/pattern_boost/`)

Thin re-export layer over `tcgraph_ext.pattern_boost`. Use this in experiments — it mirrors the style of `algorithms/word_problem.py`.

```python
from algorithms.pattern_boost.pb_tcgraph import (
    RelatorMap, SampleConfig, LocalSearchConfig, LocalSearcher,
    HellingerScorer, HammingScorer, EditScorer,
    WordProductLengthScorer, AbelianizedProductLengthScorer,
    HighPowerCountScorer, LengthRangeScorer, HybridScorer,
    BigramFrequencies, ScorerConfig,
    build_standard_relators, generate_word, generate_words,
    load_challenges, to_json, from_json,
    uniform_len, zero_heavy_len, geometric_len, constant_len,
    make_hybrid_scorer,
    score_word,   # convenience: accepts str or Word
)
```

## Key Concepts

### FactoredWord
A word represented as an ordered list of `Factor` objects. Each factor is a conjugated relator:

```
Factor = conjugator^{-1} * R[idx] * conjugator
```

Expanding a `FactoredWord` via `rm.expand(fw.factors)` yields a word that is trivial in B(2,5) by construction (product of conjugates of relators).

### RelatorMap
Maps 1-based indices to relator `Word` objects. Two constructors:
- `RelatorMap(list[Word])` — auto-names R1, R2, …
- `RelatorMap(list[tuple[str, Word]])` — named relators

In the full workflow, the RelatorMap is either **generated** (standard `build_hybrid_relators` procedure, controlled by `relators.max_relator_len` and `relators.num_generators`) or **loaded from a file** (set `relators.relators_file` to a path; names from the JSON keys are preserved).

### SampleConfig

| Field | Default | Meaning |
|-------|---------|---------|
| `min_factors` | 3 | min conjugated-relator factors |
| `max_factors` | 8 | max factors |
| `conj_len` | `uniform_len(4)` | length sampler for conjugators |
| `num_generators` | 2 | generators in B(2,5) |
| `min_expanded_len` | 10 | discard shorter expanded words |

**Conjugator length samplers:**
```python
uniform_len(max_len)                  # uniform over [0, max_len]
zero_heavy_len(max_len, zero_prob=0.3) # zero with prob zero_prob, else uniform
geometric_len(max_len, p=0.5)         # geometric(p) capped at max_len
constant_len(len)                     # always exactly len
```

### LocalSearchConfig

| Field | Default | Meaning |
|-------|---------|---------|
| `beam_width` | 1 | 1 = greedy |
| `max_steps` | 100 | search steps |
| `num_conj_samples` | 5 | conjugator mutations per factor per step |
| `num_insert_samples` | 5 | factor insertions per step |
| `conj_max_len` | 4 | max conjugator length for mutations |
| `min_factors` | 1 | don't delete below this |
| `max_factors` | 20 | don't insert above this |

### Scoring

All scorers share the same interface: challenges are baked in at construction; `scoreWord`/`scoreSample` take only the query. All scores are in `[0, 1]` (higher = more similar to at least one challenge).

| Type key | Class | Constructor inputs | Notes |
|---|---|---|---|
| `hellinger` | `HellingerScorer` | challenges, num_generators | Hellinger distance on bigram-frequency histograms |
| `hamming` | `HammingScorer` | challenges | Extended Hamming distance, normalized by max length |
| `edit` | `EditScorer` | challenges | Levenshtein edit distance, normalized by max length |
| `word_product_length` | `WordProductLengthScorer` | challenges | Free-group metric: `len(reduce(w·c⁻¹)) / (|w|+|c|)` |
| `abelianized_product_length` | `AbelianizedProductLengthScorer` | challenges, num_generators, q=5 | Distance in `Z_q^n` abelian quotient |
| `high_power_count` | `HighPowerCountScorer` | threshold=2.5 | 1.0 = no subword power > threshold; no challenges needed |
| `length_range` | `LengthRangeScorer` | min_length, max_length, sigma, reduce_before_scoring=false | Gaussian score peaked at target length range; no challenges needed; when `reduce_before_scoring=true` scores the 2.5-reduced length |
| `hybrid` | `HybridScorer` | sub-scorers + weights | Weighted sum; built via `make_hybrid_scorer` or YAML `components` |

Use `make_scorer(type, challenges, num_generators)` from `algorithms.pattern_boost.common` to build any scorer by name, or build directly from the class. For hybrid:

```python
from algorithms.pattern_boost.common import make_scorer
scorer = make_scorer("hybrid", challenge_words, num_generators=2, components=[
    {"type": "edit", "weight": 1.0},
    {"type": "abelianized_product_length", "weight": 0.5, "params": {"q": 5}},
    {"type": "high_power_count", "weight": 0.5, "params": {"threshold": 2.5}},
])
```

Sub-scorers in a hybrid always use default `ScorerConfig`; compression and `reduce2_5` are applied once by the outer `HybridScorer`.

### JSON I/O
```python
s = to_json(fw, rm)        # FactoredWord → JSON string
fw = from_json(s)          # JSON string → FactoredWord
fw = from_json(d)          # Python dict also accepted
```

JSON structure:
```json
{
  "word": [1, 2, -1, ...],
  "relator_indices": [3, 1, 2],
  "relator_names": ["R3", "R1", "R2"],
  "conjugators": [[1, -2], [], [2]],
  "version": 1
}
```

## Typical Workflow

```python
from algorithms.word_problem import word_from_str
from algorithms.pattern_boost.common import make_scorer, load_challenges
from algorithms.pattern_boost.pb_tcgraph import (
    RelatorMap, SampleConfig, LocalSearchConfig, LocalSearcher,
    generate_words, uniform_len,
)

# 1. Build RelatorMap
rm = RelatorMap([(name, word_from_str(w)) for name, w in my_relators])

# 2. Load challenges (baked into scorer at construction)
challenge_words, _ = load_challenges("data/my_challenges.json")

# 3. Build scorer
scorer = make_scorer("edit", challenge_words, num_generators=2)
# Hybrid example:
# scorer = make_scorer("hybrid", challenge_words, num_generators=2, components=[
#     {"type": "edit", "weight": 1.0},
#     {"type": "high_power_count", "weight": 0.5, "params": {"threshold": 2.5}},
# ])

# 4. Generate samples
cfg = SampleConfig()
cfg.conj_len = uniform_len(4)
samples = generate_words(rm, cfg, count=50, seed=42)

# 5. Local search
ls_cfg = LocalSearchConfig()
ls_cfg.max_steps = 100
searcher = LocalSearcher(rm, scorer, ls_cfg)
results = [searcher.search(fw, seed=42 + i) for i, fw in enumerate(samples)]
```

## Experiments

### `experiment_local_search.py`

**Goal:** Sweep over local-search settings and scorer types; save scored FactoredWords for downstream training.

**Run:**
```bash
uv run python -m experiments.pattern_boost.experiment_local_search \
  configs/experiment1_local_search.yaml
```

Cartesian product over `sweep.max_steps × sweep.beam_width × sweep.score_types`. Results written as `pb.to_json`-format JSON + sibling `.scores.json`.

---

## Entry-point scripts

All three scripts take a single positional YAML config (no `--config` flag).

### `pattern_boost_main.py` — full workflow loop
```bash
uv run python -m experiments.pattern_boost.pattern_boost_main \
  configs/pattern_boost.yaml \
  [--output-dir PATH] \
  [--init-model-from PATH]
```
- `--output-dir` overrides `output_dir`; a timestamped `run_YYYYMMDD_HHMMSS` subdir is always appended.
- `--init-model-from PATH` overrides `init_model_from` (vocab-size validated against the tokenizer at load).

### `train_from_factor_words.py` — Phase 1 supplementary
Tokenize + train + sample FactoredWords from a fixed input pool.
```bash
uv run python -m experiments.pattern_boost.train_from_factor_words \
  configs/train_from_factor_words.yaml
```
Outputs in `output_dir/`:
- `sampled_factors.json` — `pb.to_json` format (consumable by experiments downstream)
- `sampled_word_score.jsonl` — one line per sampled word: `{word, length, score}`

### `experiment_local_search.py` — sweep
See above.

---

## YAML configuration

Schema mirrors the dataclasses in `algorithms/pattern_boost/workflow.py` — `WorkflowConfig` and the nested
`RelatorsCfg`, `ChallengesCfg`, `TokenizerCfg`, `ModelConfig`, `TrainConfig`, `InitialSamplingCfg`,
`GenerationCfg`, `LocalSearchCfg`, `ScoringCfg`, `WorkflowLoopCfg`, `LoggingCfg`.

Top-level fields: `output_dir`, `seed`, `init_model_from` (optional path to a saved HF checkpoint).

Recently-added knobs worth knowing:

| Field | Effect |
|---|---|
| `challenges.sample_size` | Random subset of challenges; seeded by `cfg.seed`. Omit/null = load all. |
| `tokenizer.use_bpe` (default `false`) + `vocab_size` | BPE merges on top of structured FactoredWord tokens. |
| `workflow.train_from_scratch` | Rebuild model with fresh weights each iteration (uses `_release_model` to free GPU/MPS memory). |
| `workflow.max_pool_size` | Initial pool truncated by score (highest kept). |
| `scoring.type` | `hellinger \| hamming \| edit \| word_product_length \| abelianized_product_length \| high_power_count \| length_range \| hybrid`. |
| `scoring.components` | Required when `scoring.type: hybrid`. List of `{type, weight, params?}` dicts. `params` keys: `q` (abelianized), `threshold` (high_power_count), `min_length`/`max_length`/`sigma`/`reduce_before_scoring` (length_range). Sub-scorers always use default `ScorerConfig`. |
| `init_model_from` | Warm-start LM from a previous run's checkpoint dir (e.g. `.../iter_0009/model`). |
| `relators.relators_file` | Path to a JSON relators file (`{"R1": {"version": 1, "word": [...]}, ...}`). When set, the RelatorMap is loaded from this file instead of being generated via `build_hybrid_relators`. `relators.num_generators` is still required for the scorer and tokenizer. |

Provided configs in `configs/`:
- `pattern_boost.yaml` — full workflow (includes commented-out hybrid scoring example)
- `pattern_boost_2_5_reduced.yaml` — challenges from 2.5-reduced shortlex KB rules
- `pattern_boost_smoke.yaml` — 2-iteration smoke test
- `train_from_factor_words.yaml`
- `experiment1_local_search.yaml`
- `pattern_boost_sweep.yaml` / `pattern_boost_sweep_1.yaml` — hyperparameter sweeps
- `bpe_train.yaml`
- `generate_sample.yaml`

---

## Outputs

Under `<output_dir>/run_<timestamp>/`:

- `run.jsonl` — JSON event log. Notable events:
  - `pb_challenges_loaded` (`count` / `available` / `requested`)
  - `pb_model_built` or `pb_model_loaded_from_ckpt`
  - `pb_train_seq_lengths` (min/max/mean/p50/p90/p95/p99, `over_max_len_pct`)
  - `pb_challenge_match`, `pb_perfect_score_candidate`
- `run.matches.jsonl` — one JSON record per cyclic challenge match (filename stem matches the log file). Fields: `iteration`, `timestamp`, `challenge_name`, `challenge_word`, `candidate_word`, `factor_word`, `score`, `scorer_type`.
- `iter_NNNN/model/` — HuggingFace checkpoint per saved iteration (controlled by `workflow.save_every`). Reload via `--init-model-from <path>`.
