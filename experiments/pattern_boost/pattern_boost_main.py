# Created by: Alex Myasnikov
# Date: May 14, 2026

"""Phase 2 entry point: full PatternBoost workflow loop."""

import argparse
from pathlib import Path

from algorithms.pattern_boost.workflow import load_config, run_pattern_boost


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config', type=Path, help='YAML config path')
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Override config.output_dir')
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)

    run_pattern_boost(cfg)


if __name__ == '__main__':
    main()
