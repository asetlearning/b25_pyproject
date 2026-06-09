# Experiment Dashboards

Interactive dashboards built with [marimo](https://marimo.io) for analyzing experiment results.

## Setup

```bash
uv sync   # installs marimo, pandas, plotly
```

## Running a dashboard

**Edit mode** (live-editable cells, recommended for development):
```bash
uv run marimo edit experiments/dashboards/sweep_dashboard.py
```

**Run mode** (read-only, cleaner UI for viewing results):
```bash
uv run marimo run experiments/dashboards/sweep_dashboard.py
```

Both commands open a browser tab automatically.

---

## Dashboards

### `sweep_dashboard.py` — PatternBoost hyperparameter sweep

Analyzes results from a `pattern_boost_main.py sweep` run.

**Input:** path to sweep root dir (e.g. `output/pb_sweep/sweep_20260522_013909`)

**Metrics per parameter combo:**
- Success rate % — fraction of trials that found a challenge match
- Avg iterations to first match (matched trials only)
- Avg time in seconds to first match (matched trials only)

**Parameters swept:** `beam_width`, `max_steps`, `max_pool_size`

To analyze a different sweep, paste its path into the folder input box and press Enter.
