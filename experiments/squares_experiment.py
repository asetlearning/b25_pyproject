"""
Experiment: Detect word squares in relators.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

from algorithms.word_squares import find_squares
from algorithms.logger import get_logger, configure_logging, log_structured
from data.relators_io import load_relators, convert_to_strings

logger = get_logger(__name__)


def analyze_squares_in_relators(
    relators_file: str,
    output_file: str
) -> Dict[str, Any]:
    """
    Detect all word squares in relators from file.

    Args:
        relators_file: JSON file with relators
        output_file: Where to save results

    Returns:
        Statistics about squares found
    """
    log_structured(logger, 'info', event='loading_relators', file=relators_file)
    relators_data = load_relators(relators_file)
    relators_strings = convert_to_strings(relators_data)
    log_structured(logger, 'info', event='relators_loaded', count=len(relators_strings))

    stats = {
        "total_relators": len(relators_strings),
        "relators_with_squares": 0,
        "relators_without_squares": 0,
        "total_squares": 0,
        "max_squares_in_relator": 0,
        "min_square_length": float('inf'),
        "max_square_length": 0
    }

    results = {}

    # Process each relator with progress bar
    for rel_name, rel_data in tqdm(
        relators_strings.items(),
        desc="Analyzing relators",
        unit="relator"
    ):
        word = rel_data["word"]
        word_length = len(word)

        # Find all squares in this relator
        squares = find_squares(word)
        num_squares = len(squares)

        # Update statistics
        stats["total_squares"] += num_squares

        if num_squares > 0:
            stats["relators_with_squares"] += 1
            stats["max_squares_in_relator"] = max(
                stats["max_squares_in_relator"],
                num_squares
            )

            # Track square lengths
            square_lengths = []
            for square in squares:
                sq_len = len(square)
                square_lengths.append(sq_len)
                stats["min_square_length"] = min(stats["min_square_length"], sq_len)
                stats["max_square_length"] = max(stats["max_square_length"], sq_len)

            log_structured(
                logger, 'info',
                event='squares_found',
                relator=rel_name,
                word_length=word_length,
                num_squares=num_squares,
                square_lengths=sorted(square_lengths, reverse=True),
                squares=[s for s in sorted(squares, key=len, reverse=True)]
            )
        else:
            stats["relators_without_squares"] += 1
            log_structured(
                logger, 'info',
                event='no_squares_found',
                relator=rel_name,
                word_length=word_length
            )

        # Store results
        results[rel_name] = {
            "word": word,
            "word_length": word_length,
            "squares": sorted(list(squares), key=len, reverse=True),
            "num_squares": num_squares
        }

    # Calculate average
    if stats["total_relators"] > 0:
        stats["avg_squares_per_relator"] = round(
            stats["total_squares"] / stats["total_relators"],
            2
        )
    else:
        stats["avg_squares_per_relator"] = 0.0

    # Handle edge case where no squares found
    if stats["min_square_length"] == float('inf'):
        stats["min_square_length"] = 0

    # Prepare output
    output = {
        "relators": results,
        "statistics": stats
    }

    # Save results
    log_structured(logger, 'info', event='saving_results', file=output_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    # Log summary statistics
    log_structured(
        logger, 'info',
        event='experiment_complete',
        total_relators=stats['total_relators'],
        relators_with_squares=stats['relators_with_squares'],
        relators_without_squares=stats['relators_without_squares'],
        total_squares=stats['total_squares'],
        avg_squares_per_relator=stats['avg_squares_per_relator'],
        max_squares_in_relator=stats['max_squares_in_relator'],
        min_square_length=stats['min_square_length'],
        max_square_length=stats['max_square_length']
    )

    return stats


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect word squares in relators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python squares_experiment.py \\
    data/b25_nontrivial_relators_update_azsw1h6x_20260129.json \\
    output/squares_analysis.json
"""
    )
    parser.add_argument("relators_file", help="JSON file with relators")
    parser.add_argument("output_file", help="Where to save results")
    parser.add_argument("--log-file", help="Log to file (default: stdout)")

    args = parser.parse_args()

    # Configure logging (always JSON)
    configure_logging(args.log_file)

    analyze_squares_in_relators(args.relators_file, args.output_file)
