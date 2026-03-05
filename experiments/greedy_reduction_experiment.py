"""
Experiment: Apply greedy power reduction to all relators.

Created by: Alex Myasnikov
Date: February 26, 2026
"""

import json
import logging
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging.handlers import QueueListener
from typing import Dict, Any

from tqdm import tqdm

from algorithms.greedy_reduction import greedy_reduce
from algorithms.logger import get_logger, configure_logging, configure_logging_queue, log_structured, JsonFormatter
from data.relators_io import load_relators, convert_to_strings, convert_to_integers

logger = get_logger(__name__)


def _reduce_one(rel_name: str, rel_data: dict) -> dict:
    """Reduce a single relator. Module-level for picklability."""
    original_word = rel_data["word"]
    original_len = len(original_word)
    log_structured(logger, 'info', event='processing_relator', relator=rel_name)
    try:
        reduced_word = greedy_reduce(original_word, rel_name)
        length_diff = original_len - len(reduced_word)
        if length_diff > 0:
            log_structured(logger, 'info', event='relator_shortened',
                           relator=rel_name,
                           original_length=original_len,
                           new_length=len(reduced_word),
                           reduction=length_diff)
        return {
            "rel_name": rel_name,
            "version": rel_data["version"],
            "original_len": original_len,
            "reduced_word": reduced_word,
            "error": None,
        }
    except Exception as e:
        log_structured(logger, 'error', event='error_reducing_relator',
                       relator=rel_name, error=str(e))
        return {
            "rel_name": rel_name,
            "version": rel_data["version"],
            "original_len": original_len,
            "reduced_word": original_word,
            "error": str(e),
        }


def _process_batch(batch: list, queue) -> list:
    """Process a batch of (rel_name, rel_data) pairs. Configures logging for worker process."""
    configure_logging_queue(queue)
    return [_reduce_one(rel_name, rel_data) for rel_name, rel_data in batch]


def _aggregate_results(raw_results: list, relators_data: dict) -> tuple:
    """Build reduced_strings, stats, details from flat result list."""
    reduced_strings = {}
    stats = {
        "total_relators": len(raw_results),
        "shortened": 0,
        "unchanged": 0,
        "total_length_reduction": 0,
    }
    details = {}

    for r in raw_results:
        rel_name = r["rel_name"]
        if r["error"] is not None:
            continue

        length_diff = r["original_len"] - len(r["reduced_word"])
        if length_diff > 0:
            stats["shortened"] += 1
            stats["total_length_reduction"] += length_diff
            details[rel_name] = "greedy power reduction"
        else:
            stats["unchanged"] += 1

        reduced_strings[rel_name] = {
            "version": r["version"],
            "word": r["reduced_word"],
        }

    return reduced_strings, stats, details


def _save_output(reduced_strings: dict, relators_data: dict,
                 details: dict, output_file: str) -> None:
    result = convert_to_integers(reduced_strings, relators_data, details=details)
    log_structured(logger, 'info', event='saving_results', file=output_file)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)


def run_greedy_reduction(relators_file: str, output_file: str) -> Dict[str, Any]:
    """Apply greedy power reduction to all relators sequentially.

    Args:
        relators_file: JSON file with relators
        output_file: Where to save results

    Returns:
        Stats: total_relators, shortened, unchanged, total_length_reduction
    """
    log_structured(logger, 'info', event='loading_relators', file=relators_file)
    relators_data = load_relators(relators_file)
    relators_strings = convert_to_strings(relators_data)

    raw_results = []
    for rel_name, rel_data in relators_strings.items():
        raw_results.append(_reduce_one(rel_name, rel_data))

    reduced_strings, stats, details = _aggregate_results(raw_results, relators_data)
    _save_output(reduced_strings, relators_data, details, output_file)

    log_structured(logger, 'info', event='experiment_complete',
                   total_relators=stats['total_relators'],
                   shortened=stats['shortened'],
                   unchanged=stats['unchanged'],
                   total_length_reduction=stats['total_length_reduction'])

    return stats


def run_greedy_reduction_parallel(
    relators_file: str,
    output_file: str,
    workers: int,
    log_file: str | None = None,
) -> Dict[str, Any]:
    """Apply greedy power reduction using multiple processes.

    Args:
        relators_file: JSON file with relators
        output_file: Where to save results
        workers: Number of worker processes

    Returns:
        Stats: total_relators, shortened, unchanged, total_length_reduction
    """
    log_structured(logger, 'info', event='loading_relators', file=relators_file)
    relators_data = load_relators(relators_file)
    relators_strings = convert_to_strings(relators_data)

    items = list(relators_strings.items())
    batch_size = math.ceil(len(items) / workers)
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    queue = multiprocessing.Manager().Queue(-1)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(JsonFormatter())
        listener = QueueListener(queue, file_handler, respect_handler_level=True)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(JsonFormatter())
        listener = QueueListener(queue, stream_handler, respect_handler_level=True)

    listener.start()
    raw_results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_batch, batch, queue): len(batch)
                       for batch in batches}
            with tqdm(total=len(items), desc="reducing relators") as pbar:
                for future in as_completed(futures):
                    batch_results = future.result()
                    raw_results.extend(batch_results)
                    pbar.update(futures[future])
    finally:
        listener.stop()

    reduced_strings, stats, details = _aggregate_results(raw_results, relators_data)
    _save_output(reduced_strings, relators_data, details, output_file)

    log_structured(logger, 'info', event='experiment_complete',
                   total_relators=stats['total_relators'],
                   shortened=stats['shortened'],
                   unchanged=stats['unchanged'],
                   total_length_reduction=stats['total_length_reduction'])

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Greedy power reduction on relators")
    parser.add_argument("relators_file", help="JSON file with relators")
    parser.add_argument("output_file", help="Where to save results")
    parser.add_argument("--log-file", help="Log to file (default: stdout)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (default: 1)")

    args = parser.parse_args()

    if args.workers > 1:
        configure_logging()  # main process logs to stdout; listener owns the file
        run_greedy_reduction_parallel(args.relators_file, args.output_file, args.workers, args.log_file)
    else:
        configure_logging(args.log_file)
        run_greedy_reduction(args.relators_file, args.output_file)
