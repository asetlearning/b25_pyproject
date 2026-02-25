# Burnside Problem Experimentation

Python project for combinatorial group theory experiments related to the Burnside problem.

## Overview

The Burnside problem asks whether a finitely generated group in which every element has finite order must be finite. This project provides tools for experimenting with word processing and group-theoretic computations.

## Structure

```
├── data/          # Data sources and experimental results
├── experiments/   # Experiment implementations
├── algorithms/    # Core algorithms for word processing
└── requirements.txt
```

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install project in editable mode
pip install -e .
```

## Testing

Run tests:

```bash
python3 -m pytest tests/ -v
```

## Usage

### Rewrite Rules Experiment

Apply Knuth-Bendix rewrite rules to relators:

```bash
python3 experiments/rewrite_experiment.py \
  data/b25_nontrivial_relators_update_azsw1h6x_20260129.json \
  data/kb_mixer_data/shortlex \
  output/rewritten_relators.json
```

Logs output to stdout as JSON. Add `--log-file output/experiment.log` to write to file instead.

**JSON log example:**
```json
{"timestamp": "2026-02-10T14:32:15.123456", "level": "INFO", "logger": "experiments.rewrite_experiment", "event": "relator_rewritten", "relator": "R1", "original_length": 100, "new_length": 85, "reduction": 15, "result": "shortened"}
```

### Word Squares Detection Experiment

Detect all word squares (W² = WW) in relators:

```bash
python3 experiments/squares_experiment.py \
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
