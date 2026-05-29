# Burnside Problem Experimentation

Python project for combinatorial group theory experiments related to the Burnside problem.

## Overview

The Burnside problem asks whether a finitely generated group in which every element has finite order must be finite. This project provides tools for experimenting with word processing and group-theoretic computations.

## Structure

```
├── algorithms/    # Core algorithms for word processing
├── cpp/           # C++ pybind11 extension (tcgraph bindings)
│   ├── tcgraph/   # git submodule: Todd-Coxeter graph library
│   ├── bindings.cpp
│   └── CMakeLists.txt
├── data/          # Data sources and experimental results
├── experiments/   # Experiment scripts
├── ml/            # Machine learning components (BPE tokenizer)
├── models/        # Saved model files
└── tests/         # Unit tests
```

## Setup

### Prerequisites

The C++ extension requires cmake and a C++20 compiler:

```bash
# Ubuntu/Debian
sudo apt-get install cmake g++ ninja-build
sudo apt-get install python3.12-dev

# macOS
brew install cmake ninja
```

### Install

Uses [uv](https://docs.astral.sh/uv/) for env/dep management.

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd b25_pyproject

# Or, if already cloned:
git submodule update --init --recursive

# Install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv, install deps, build C++ extension
uv sync
```

## Testing

Run tests:

```bash
uv run pytest tests/ -v
```

Verify words are trivial in $\bar{B}(2,5)$ using **gap**:

from the folder with gap script and relators file:
```bash
gap docker run -it -v "$PWD:/work" gap bash
cd /work/
gap -b -q -c 'json_file := "b25_challenge_original_2.5reduced_cyclic4_20260402.json"; Read("verify_b25_relators.gap");'
```


## Usage

### Word Problem (tcgraph C++ extension)

Test whether a word is trivial in B(2,5) using Todd-Coxeter approximation:

```python
from algorithms.word_problem import Word, construct_cayley_approximation

# Generators: 1=a, 2=b, -1=A (a⁻¹), -2=B (b⁻¹)
w = Word([1, 2, -1, -2])   # commutator [a,b]
result = construct_cayley_approximation(w)
# True  → word is trivial in B(2,5)
# False → word is non-trivial
# None  → inconclusive (coset bound reached)

# Optional: raise coset bound (default 20_000_000)
result = construct_cayley_approximation(w, coset_bound=50_000_000)
```

Word operations:
```python
w = Word([1, 2])       # ab
v = w * Word([-1])     # abA
u = -w                 # BA  (inverse)
p = w ^ 3              # ababab
w.reduce()             # free reduction
len(w)                 # length
w.get_letters()        # list of ints
```

### Rewrite Rules Experiment

Apply Knuth-Bendix rewrite rules to relators:

```bash
uv run python experiments/rewrite_experiment.py \
  data/b25_nontrivial_relators_update_azsw1h6x_20260129.json \
  data/kb_mixer_data/shortlex \
  output/rewritten_relators.json
```

Logs output to stdout as JSON. Add `--log-file output/experiment.log` to write to file instead.

**JSON log example:**
```json
{"timestamp": "2026-02-10T14:32:15.123456", "level": "INFO", "logger": "experiments.rewrite_experiment", "event": "relator_rewritten", "relator": "R1", "original_length": 100, "new_length": 85, "reduction": 15, "result": "shortened"}
```

### Greedy Power Reduction Experiment

Apply greedy power reduction to all relators:

```bash
uv run python experiments/greedy_reduction_experiment.py \
  data/b25_nontrivial_relators_update_azsw1h6x_20260129.json \
  output.json
```

Use `--workers N` to run in parallel across N processes:

```bash
uv run python experiments/greedy_reduction_experiment.py \
  data/b25_nontrivial_relators_update_azsw1h6x_20260129.json \
  output.json --workers 4
```

Add `--log-file output/experiment.log` to write logs to file instead of stdout.

### BPE Tokenizer Training Experiment

Train a Byte-Pair Encoding tokenizer on all relator words:

```bash
uv run python experiments/bpe_train_experiment.py \
  data/b25_nontrivial_relators_update_azsw1h6x_20260129.json
```

By default the model is saved to `models/tokenizer/bpe_model.json` with a vocabulary size of 1000. Both can be overridden:

```bash
uv run python experiments/bpe_train_experiment.py \
  data/b25_nontrivial_relators_update_azsw1h6x_20260129.json \
  --output models/tokenizer/my_model.json \
  --vocab-size 2000
```

The saved model can be loaded back with `BPETokenizerWords.from_file(path)`:

```python
from ml.bpe import BPETokenizerWords

tokenizer = BPETokenizerWords.from_file("models/tokenizer/bpe_model.json")
token_ids = tokenizer.encode("abABabAB")
word = tokenizer.decode(token_ids)
```

### Word Squares Detection Experiment

Detect all word squares (W² = WW) in relators:

```bash
uv run python experiments/squares_experiment.py \
  data/b25_nontrivial_relators_update_azsw1h6x_20260129.json \
  output/squares_analysis.json
```

Logs output to stdout as JSON. Add `--log-file output/experiment.log` to write to file instead.

**JSON log example:**
```json
{"timestamp": "2026-02-10T14:32:15.123456", "level": "INFO", "logger": "experiments.squares_experiment", "event": "squares_found", "relator": "R1", "word_length": 100, "num_squares": 3, "square_lengths": [20, 18, 12], "squares": ["ABabABab", "BaBbBaBb", "AbAb"]}
```

This analyzes each relator to find all subwords that are squares (concatenation of a word with itself). Results include:
- List of squares found in each relator
- Statistics: total squares, relators with/without squares, square length distribution
- Detailed logging of each square detected

### PatternBoost Workflow

LM-guided local search for B(2,5): a GPT2 model is trained on FactoredWord sequences, sampled candidates are improved with beam-search local search, and the top-scored survivors (by bigram similarity to challenge words) are fed back into the next training round. Optional cyclic-match stopping criterion writes hits to `run.matches.jsonl`.

All three entry points take a positional YAML config:

```bash
# Full workflow loop
uv run python -m experiments.pattern_boost.pattern_boost_main \
  configs/pattern_boost.yaml [--output-dir PATH] [--init-model-from PATH]

# Train on a fixed FactoredWord pool, sample, score
uv run python -m experiments.pattern_boost.train_from_factor_words \
  configs/train_from_factor_words.yaml

# Local-search sweep over (max_steps × beam_width × score_types)
uv run python -m experiments.pattern_boost.experiment_local_search \
  configs/experiment1_local_search.yaml
```

See [`experiments/pattern_boost/README.md`](experiments/pattern_boost/README.md) for the full config schema and output layout (`run.jsonl`, `run.matches.jsonl`, `iter_NNNN/model/`).

### Structured Logging API

All logs output as JSON with custom fields:

```python
from algorithms.logger import get_logger, log_structured

logger = get_logger(__name__)

# Option 1: Using extra parameter
logger.info("Processing complete", extra={
    'relator': 'R1',
    'processing_time': 1.23,
    'success': True
})

# Option 2: Using log_structured helper (recommended)
log_structured(
    logger, 'info',
    event='processing_complete',
    relator='R1',
    processing_time=1.23,
    success=True
)
```

Both produce same JSON output:
```json
{"timestamp": "2026-02-10T14:32:15.123456", "level": "INFO", "logger": "my_module", "event": "processing_complete", "relator": "R1", "processing_time": 1.23, "success": true}
```

## Docker Usage

### Build Container

```bash
docker-compose build
```

### Run Experiments

**Rewrite experiment:**
```bash
docker-compose run --rm b25-experiments \
  experiments/rewrite_experiment.py \
  data/b25_nontrivial_relators_update_azsw1h6x_20260129.json \
  data/kb_mixer_data/shortlex \
  output/rewritten_relators.json
```

**Squares experiment:**
```bash
docker-compose run --rm b25-experiments \
  experiments/squares_experiment.py \
  data/b25_nontrivial_relators_update_azsw1h6x_20260129.json \
  output/squares_analysis.json
```

**With file logging:**
```bash
docker-compose run --rm b25-experiments \
  experiments/rewrite_experiment.py \
  data/b25_nontrivial_relators_update_azsw1h6x_20260129.json \
  data/kb_mixer_data/shortlex \
  output/rewritten_relators.json \
  --log-file output/experiment.log
```

### Alternative: Direct docker run

```bash
# Build
docker build -t b25-experiments .

# Run with volume mounts
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/output:/app/output" \
  b25-experiments \
  experiments/rewrite_experiment.py \
  data/b25_nontrivial_relators_update_azsw1h6x_20260129.json \
  data/kb_mixer_data/shortlex \
  output/rewritten_relators.json
```
