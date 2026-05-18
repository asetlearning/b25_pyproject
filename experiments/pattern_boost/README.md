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
| `pb_scoring.h/cpp` | `BigramFrequencies`, `HellingerScorer`, `loadChallenges` |
| `pb_local_search.h/cpp` | `LocalSearchConfig`, `LocalSearcher` |

### Python bindings (`tcgraph_ext.pattern_boost`)

Exposed as a submodule of `tcgraph_ext` (same `.so` as the rest of the C++ interface). Built in `cpp/bindings_pattern_boost.cpp`, registered via `init_pattern_boost()` called from `cpp/bindings.cpp`.

### Python wrapper (`algorithms/pattern_boost/`)

Thin re-export layer over `tcgraph_ext.pattern_boost`. Use this in experiments — it mirrors the style of `algorithms/word_problem.py`.

```python
from algorithms.pattern_boost import (
    RelatorMap, SampleConfig, LocalSearchConfig, LocalSearcher,
    HellingerScorer, BigramFrequencies,
    build_standard_relators, generate_word, generate_words,
    load_challenges, to_json, from_json,
    uniform_len, zero_heavy_len, geometric_len, constant_len,
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
`HellingerScorer.score_sample(fw, rm, challenge_freqs)` returns a score in `[0, 1]`:
- `1.0` = perfect bigram match to some challenge word
- `0.0` = maximally dissimilar to all challenges

Score is `max over challenges of (1 - hellinger_distance(bigram(expanded), bigram(challenge)))`.

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
from algorithms.pattern_boost import (
    RelatorMap, SampleConfig, LocalSearchConfig, LocalSearcher,
    BigramFrequencies, HellingerScorer, generate_words, uniform_len,
)

# 1. Build RelatorMap
rm = RelatorMap([(name, word_from_str(w)) for name, w in my_relators])

# 2. Build challenge frequencies
challenge_freqs = [BigramFrequencies(word_from_str(w), 2) for w in challenge_words]

# 3. Generate samples
cfg = SampleConfig()
cfg.conj_len = uniform_len(4)
samples = generate_words(rm, cfg, count=50, seed=42)

# 4. Score before search
scorer = HellingerScorer()
before = [scorer.score_sample(fw, rm, challenge_freqs) for fw in samples]

# 5. Local search
ls_cfg = LocalSearchConfig()
ls_cfg.max_steps = 100
searcher = LocalSearcher(rm, challenge_freqs, ls_cfg)
results = [searcher.search(fw, seed=42 + i) for i, fw in enumerate(samples)]
```

## Experiments

### `experiment1_local_search.py`

**Goal:** Verify that local search improves scores, and check whether any generated word happens to match a known shortlex relator.

**Setup:**
- **RelatorMap** — 68 shortlex relators from `reduced_rules.json`
- **Challenge words** — same relators (bigram distributions as targets)
- **Samples** — 20 random trivial words (`min_factors=2`, `max_factors=6`)
- **Local search** — greedy, 50 steps

**Run:**
```bash
uv run python experiments/pattern_boost/experiment_local_search.py
```

**Output:** per-sample before/after score table, aggregate stats, exact-match check, top-3 results.
