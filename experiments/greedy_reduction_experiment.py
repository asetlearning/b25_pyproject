"""
Experiment: Apply greedy power reduction to all relators.

Created by: Alex Myasnikov
Date: February 26, 2026
"""

import json
from typing import Dict, Any

from algorithms.greedy_reduction import greedy_reduce
from algorithms.logger import get_logger, configure_logging, log_structured
from data.relators_io import load_relators, convert_to_strings, convert_to_integers

logger = get_logger(__name__)


def run_greedy_reduction(relators_file: str, output_file: str) -> Dict[str, Any]:
    """Apply greedy power reduction to all relators.

    Args:
        relators_file: JSON file with relators
        output_file: Where to save results

    Returns:
        Stats: total_relators, shortened, unchanged, total_length_reduction
    """
    log_structured(logger, 'info', event='loading_relators', file=relators_file)
    relators_data = load_relators(relators_file)
    relators_strings = convert_to_strings(relators_data)

    stats = {
        "total_relators": len(relators_strings),
        "shortened": 0,
        "unchanged": 0,
        "total_length_reduction": 0,
    }

    reduced_strings = {}
    details = {}

    do_skip = True
    for rel_name, rel_data in relators_strings.items():

        # if rel_name == "R103":
        #     do_skip = False
        #
        # if do_skip:
        #     continue


        original_word = rel_data["word"]
        original_len = len(original_word)

        log_structured(logger, 'info', event='processing_relator', relator=rel_name)

        try:
            reduced_word = greedy_reduce(original_word, rel_name)
            new_len = len(reduced_word)

            length_diff = original_len - new_len

            if length_diff > 0:
                stats["shortened"] += 1
                stats["total_length_reduction"] += length_diff
                details[rel_name] = "greedy power reduction"
                log_structured(
                    logger, 'info',
                    event='relator_shortened',
                    relator=rel_name,
                    original_length=original_len,
                    new_length=new_len,
                    reduction=length_diff,
                )
            else:
                stats["unchanged"] += 1

            reduced_strings[rel_name] = {
                "version": rel_data["version"],
                "word": reduced_word,
            }
        except Exception as e:
            log_structured(logger, 'error', event='error_reducing_relator', relator=rel_name, error=str(e))

    result = convert_to_integers(reduced_strings, relators_data, details=details)

    log_structured(logger, 'info', event='saving_results', file=output_file)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    log_structured(
        logger, 'info',
        event='experiment_complete',
        total_relators=stats['total_relators'],
        shortened=stats['shortened'],
        unchanged=stats['unchanged'],
        total_length_reduction=stats['total_length_reduction'],
    )

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Greedy power reduction on relators")
    parser.add_argument("relators_file", help="JSON file with relators")
    parser.add_argument("output_file", help="Where to save results")
    parser.add_argument("--log-file", help="Log to file (default: stdout)")

    args = parser.parse_args()

    configure_logging(args.log_file)

    run_greedy_reduction(args.relators_file, args.output_file)
