"""
Experiment: Train BPE tokenizer on all relator words.

Created by: Alex Myasnikov
Date: March 7, 2026
"""

import os
import argparse

from data.relators_io import load_relators, convert_to_strings
from ml.bpe import BPETokenizerWords

ALPHABET = ['a', 'b', 'A', 'B']
DEFAULT_VOCAB_SIZE = 1000
DEFAULT_OUTPUT = "models/tokenizer/bpe_model.json"


def train_tokenizer(relators_file: str, output_path: str, vocab_size: int) -> None:
    relators_data = load_relators(relators_file)
    relators_strings = convert_to_strings(relators_data)

    # Extract word strings (values of "word" key)
    word_list = [rel["word"] for rel in relators_strings.values()]

    tokenizer = BPETokenizerWords(ALPHABET)
    tokenizer.train(word_list, vocab_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"Saved tokenizer to {output_path} (vocab size: {len(tokenizer.vocab)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on relators")
    parser.add_argument("relators_file", help="JSON file with relators")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output path for model (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
                        help=f"BPE vocabulary size (default: {DEFAULT_VOCAB_SIZE})")
    args = parser.parse_args()

    train_tokenizer(args.relators_file, args.output, args.vocab_size)
