"""
Experiment: Apply KB rewrite rules to relators.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

import json
from pathlib import Path
from typing import Dict, Any

from algorithms.rewrite_rules import parse_rules_file, apply_rules
from algorithms.word_conversion import word_to_string, string_to_word
from algorithms.logger import get_logger, configure_logging, log_structured
from data.relators_io import load_relators, convert_to_strings, convert_to_integers

logger = get_logger(__name__)


def run_rewrite_experiment(
    relators_file: str,
    rules_file: str,
    output_file: str
) -> Dict[str, Any]:
    """
    Apply rewrite rules from rules_file to all relators.

    Args:
        relators_file: JSON file with relators
        rules_file: File with rewrite rules
        output_file: Where to save results

    Returns:
        Statistics about rewrite process
    """
    log_structured(logger, 'info', event='loading_relators', file=relators_file)
    relators_data = load_relators(relators_file)
    relators_strings = convert_to_strings(relators_data)

    log_structured(logger, 'info', event='loading_rules', file=rules_file)
    rules = parse_rules_file(rules_file)
    log_structured(logger, 'info', event='rules_loaded', count=len(rules))

    stats = {
        "total_relators": len(relators_strings),
        "shortened": 0,
        "unchanged": 0,
        "lengthened": 0,
        "total_length_reduction": 0
    }

    rewritten_strings = {}

    for rel_name, rel_data in relators_strings.items():

        log_structured(logger, 'info', event='processing_relator', relator=rel_name)
        original_word = rel_data["word"]
        original_len = len(original_word)

        # Apply all rules
        rewritten_word = apply_rules(original_word, rules, progress_desc=f"Applying rules to {rel_name}")
        new_len = len(rewritten_word)

        length_diff = original_len - new_len

        if length_diff > 0:
            stats["shortened"] += 1
            stats["total_length_reduction"] += length_diff
            log_structured(
                logger, 'info',
                event='relator_rewritten',
                relator=rel_name,
                original_length=original_len,
                new_length=new_len,
                reduction=length_diff,
                result='shortened'
            )
        elif length_diff < 0:
            stats["lengthened"] += 1
            log_structured(
                logger, 'info',
                event='relator_rewritten',
                relator=rel_name,
                original_length=original_len,
                new_length=new_len,
                reduction=length_diff,
                result='lengthened'
            )
        else:
            stats["unchanged"] += 1

        rewritten_strings[rel_name] = {
            "version": rel_data.get("version", 1),
            "word": rewritten_word
        }

    # Convert back to integers with version increment for shortened words
    details = {
        name: "KB rewrite rules applied"
        for name, data in rewritten_strings.items()
        if len(data["word"]) < len(relators_strings[name]["word"])
    }

    result = convert_to_integers(rewritten_strings, relators_data, details=details)

    # Save results
    log_structured(logger, 'info', event='saving_results', file=output_file)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    log_structured(
        logger, 'info',
        event='experiment_complete',
        total_relators=stats['total_relators'],
        shortened=stats['shortened'],
        unchanged=stats['unchanged'],
        lengthened=stats['lengthened'],
        total_length_reduction=stats['total_length_reduction']
    )

    return stats


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Apply KB rewrite rules to relators")
    parser.add_argument("relators_file", help="JSON file with relators")
    parser.add_argument("rules_file", help="File with rewrite rules")
    parser.add_argument("output_file", help="Where to save results")
    parser.add_argument("--log-file", help="Log to file (default: stdout)")

    args = parser.parse_args()

    # Configure logging (always JSON)
    configure_logging(args.log_file)

    run_rewrite_experiment(args.relators_file, args.rules_file, args.output_file)
