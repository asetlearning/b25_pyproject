"""
Experiment: Apply KB rewrite rules to relators.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

import json
from pathlib import Path
from typing import Dict, Any

from jupyter_client.session import new_id

from algorithms.rewrite_rules import parse_rules_file, apply_rule
from algorithms.word_conversion import word_to_string, string_to_word, cyclically_reduce
from algorithms.logger import get_logger, configure_logging, log_structured, add_log_level_arg
from algorithms.word_problem import greedy_reduce2_5, greedy_reduce_cyclic_exhaustive2_5
from data.relators_io import load_relators, convert_to_strings, convert_to_integers

logger = get_logger(__name__)


def reduce_relator(word: str, rules: list, name: str | None = None) -> str:
    """Reduce word by interleaving rule application with greedy 2.5-power reduction.

    Applies rules one by one; on first length reduction, runs greedy_reduce_cyclic_exhaustive2_5
    and restarts from rule 1. Stops when no rule reduces the word.
    """
    current = cyclically_reduce(word)
    while True:
        for rule in rules:
            new_word, _ = apply_rule(current, rule)
            if len(new_word) < len(current):
                log_structured(logger, 'info', event='rule_reduce_length', rel_name=name, old_length=len(current),
                               new_length=len(new_word), rule=str(rule))
                current = greedy_reduce2_5(new_word) # greedy_reduce_cyclic_exhaustive2_5(new_word, [0])
                assert "aaa" not in current and "AAA" not in current and "bbb" not in current and "BBB" not in current, f"Triples in a reduced word: {current}"
                break
            # else:
            #     current = new_word
        else:
            break

    return greedy_reduce2_5(current) # Uncomment to explore more reductions. Not efficient for long words! greedy_reduce_cyclic_exhaustive2_5(current, None)


def run_rewrite_experiment(
    relators_file: str,
    rules_file: str,
    output_file: str,
    spec_relators: list[str] | None = None,
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

    if spec_relators:
        extract_relators = {}
        for rel_name, rel_data in relators_data.items():
            if rel_name in spec_relators:
                extract_relators[rel_name] = rel_data
        relators_data = extract_relators

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
        rewritten_word = reduce_relator(original_word, rules, rel_name)
        assert "aaa" not in rewritten_word and "AAA" not in rewritten_word and "bbb" not in rewritten_word and "BBB" not in rewritten_word, f"Triples in a reduced word: {rewritten_word}"
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

    add_log_level_arg(parser)

    args = parser.parse_args()

    # Configure logging (always JSON)
    configure_logging(args.log_file, args.log_level)

    run_rewrite_experiment(args.relators_file, args.rules_file, args.output_file)
