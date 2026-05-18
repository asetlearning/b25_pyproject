"""
Experiment: Train BPE tokenizer on all relator words.

Created by: Alex Myasnikov
Date: March 7, 2026

Usage:
    python -m experiments.bpe_train_experiment <config.yaml>

See configs/bpe_train.yaml for the schema.
"""

import argparse
import logging
import os
from pathlib import Path

import yaml

from algorithms.logger import configure_logging, log_structured
from data.relators_io import load_relators, convert_to_strings
from ml.bpe import BPETokenizerWords

DEFAULT_CONFIG = Path('configs/bpe_train.yaml')


def train_tokenizer(cfg: dict, logger: logging.Logger) -> None:
    relators_file = cfg['relators_file']
    output_path = cfg['output']
    vocab_size = cfg['vocab_size']
    alphabet = cfg['alphabet']

    relators_data = load_relators(relators_file)
    relators_strings = convert_to_strings(relators_data)
    word_list = [rel['word'] for rel in relators_strings.values()]
    log_structured(logger, 'info', event='bpe_train_corpus_loaded',
                   relators=len(word_list))

    tokenizer = BPETokenizerWords(alphabet)
    tokenizer.train(word_list, vocab_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    log_structured(logger, 'info', event='bpe_train_done',
                   output=output_path, vocab_size=len(tokenizer.vocab))
    print(f"Saved tokenizer to {output_path} (vocab size: {len(tokenizer.vocab)})")


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer on relators')
    parser.add_argument('config', type=Path, nargs='?', default=DEFAULT_CONFIG,
                        help=f'YAML config path (default: {DEFAULT_CONFIG})')
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    configure_logging(level='info')
    logger = logging.getLogger('bpe_train_experiment')
    log_structured(logger, 'info', event='bpe_train_start',
                   config_path=str(args.config), config=cfg)

    train_tokenizer(cfg, logger)


if __name__ == '__main__':
    main()
