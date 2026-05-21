# Created by: Alex Myasnikov
# Date: May 14, 2026

"""PatternBoost entry point — two commands: run (single) and sweep (multi-run)."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from algorithms.pattern_boost.workflow import load_config, run_pattern_boost
from algorithms.pattern_boost.sweep import load_sweep_config, run_sweep

app = typer.Typer(add_completion=False)


@app.command('run')
def cmd_run(
    config: Path = typer.Argument(..., help='YAML config path'),
    output_dir: Optional[Path] = typer.Option(None, '--output-dir', help='Override config.output_dir'),
    init_model_from: Optional[Path] = typer.Option(
        None, '--init-model-from',
        help='Override config.init_model_from (path to saved model checkpoint dir)',
    ),
):
    """Run a single PatternBoost workflow loop."""
    cfg = load_config(config)
    if output_dir is not None:
        cfg.output_dir = str(output_dir)
    if init_model_from is not None:
        cfg.init_model_from = str(init_model_from)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg.output_dir = f'{cfg.output_dir}/run_{timestamp}'
    run_pattern_boost(cfg)


@app.command('sweep')
def cmd_sweep(
    config: Path = typer.Argument(..., help='Sweep YAML config path'),
    output_dir: Optional[Path] = typer.Option(None, '--output-dir', help='Override sweep output_dir'),
):
    """Run a hyperparameter sweep (cartesian product of params) over PatternBoost."""
    cfg = load_sweep_config(config)
    if output_dir is not None:
        cfg.output_dir = str(output_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg.output_dir = f'{cfg.output_dir}/sweep_{timestamp}'
    run_sweep(cfg)


if __name__ == '__main__':
    app()
