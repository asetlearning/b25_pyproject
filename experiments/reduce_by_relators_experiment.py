"""
Experiment: Apply greedy power reduction to all relators.

Created by: Alex Myasnikov
Date: March 10, 2026
"""

import faulthandler

faulthandler.enable()

from ml.bpe import BPETokenizerWords
from algorithms.logger import get_logger
from data.relators_io import load_relators, convert_to_strings
from algorithms.word_problem import Word, attempt_to_reduce_using_relations_greedy

logger = get_logger(__name__)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Greedy power reduction on relators")
    parser.add_argument("relators_file", help="JSON file with relators")
    parser.add_argument("tokenizer", help="BPE tokenizer model")
    parser.add_argument("--log-file", help="Log to file (default: stdout)")

    args = parser.parse_args()

    # tokenizer = BPETokenizerWords.from_file("../b25_pyproject/models/tokenizer/bpe_tokenizer_b25_nontrivial_relators_update_azsw1h6x_20260129.json")
    # relators = convert_to_strings(load_relators("../b25_pyproject/data/reduce_words_power_2.5/reduced_relators_20260305.json"))
    tokenizer = BPETokenizerWords.from_file(args.tokenizer)
    relators = convert_to_strings(load_relators(args.relators_file))

    rel_to_reduce = relators["R5"]["word"]
    reduction_rels = [tokenizer.decode([t]) * 5 for t in set(tokenizer.encode(rel_to_reduce))]
    print(reduction_rels)
    attempt_to_reduce_using_relations_greedy(rel_to_reduce, reduction_rels, "R5")
