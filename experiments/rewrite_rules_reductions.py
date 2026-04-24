"""
Read rewrite rules and reduce LHS and RHS independently by removing 2.5 powers

Created by: Alex Myasnikov
Date: April 1, 2026
"""

import argparse
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from algorithms.logger import (
    configure_logging,
    configure_logging_queue,
    get_logger,
    log_structured,
    JsonFormatter
)
from algorithms.rewrite_rules import iter_rules_file
from algorithms.word_problem import greedy_reduce2_5

logger = get_logger(__name__)


def _reduce_one(idx: int, rule_str: str, lhs: str, rhs: str) -> dict:
    """Reduce LHS and RHS of a single rule independently. Module-level for picklability."""
    log_structured(logger, "info", event="processing_rule", rule=rule_str,
                   lhs_len=len(lhs), rhs_len=len(rhs))
    try:
        reduced_lhs = greedy_reduce2_5(lhs)
        reduced_rhs = greedy_reduce2_5(rhs)
        log_structured(
            logger, "info",
            event="rule_result",
            rule=rule_str,
            reduced_lhs=reduced_lhs,
            reduced_rhs=reduced_rhs,
            lhs_reduction=len(lhs) - len(reduced_lhs),
            rhs_reduction=len(rhs) - len(reduced_rhs),
        )
        return {
            "rule": rule_str,
            "original_lhs": lhs,
            "original_rhs": rhs,
            "reduced_lhs": reduced_lhs,
            "reduced_rhs": reduced_rhs,
            "error": None,
        }
    except Exception as e:
        log_structured(logger, "error", event="reduction_error", rule=rule_str, error=str(e))
        return {
            "rule": rule_str,
            "original_lhs": lhs,
            "original_rhs": rhs,
            "reduced_lhs": lhs,
            "reduced_rhs": rhs,
            "error": str(e),
        }


def _process_batch(batch: list, queue) -> list:
    """Process a batch of (idx, rule_str, lhs, rhs) tuples in a worker process."""
    configure_logging_queue(queue)
    return [_reduce_one(idx, rule_str, lhs, rhs) for idx, rule_str, lhs, rhs in batch]


def _write_results(results: list, out_f, stats: dict, seen: set) -> None:
    for r in results:
        stats["total"] += 1
        if r["error"] is not None:
            stats["errors"] += 1
            continue
        key = (r["reduced_lhs"], r["reduced_rhs"])
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        out_f.write(json.dumps({
            "rule": r["rule"],
            "original_lhs": r["original_lhs"],
            "original_rhs": r["original_rhs"],
            "reduced_lhs": r["reduced_lhs"],
            "reduced_rhs": r["reduced_rhs"],
            "reduced_rule": f"{r['reduced_lhs']}->{r['reduced_rhs']}",
        }) + "\n")
        out_f.flush()
        stats["saved"] += 1


def run_sequential(input_file: str, output_file: str) -> dict:
    stats = {"total": 0, "saved": 0, "duplicates": 0, "errors": 0}
    seen: set = set()
    items = [(idx, rule) for idx, rule in enumerate(iter_rules_file(input_file))]

    with open(output_file, "w") as out_f:
        for idx, rule in tqdm(items, desc="Reducing rules"):
            result = _reduce_one(idx, str(rule), rule.lhs, rule.rhs)
            _write_results([result], out_f, stats, seen)

    log_structured(logger, "info", event="done", **stats)
    return stats


def run_parallel(
    input_file: str,
    output_file: str,
    workers: int,
    log_file: str | None = None,
    chunk_size: int = 50,
) -> dict:
    items = [(idx, str(rule), rule.lhs, rule.rhs) for idx, rule in enumerate(iter_rules_file(input_file))]
    batches = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    queue = multiprocessing.Manager().Queue(-1)

    if log_file:
        handler = logging.FileHandler(log_file, mode="w")
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    from logging.handlers import QueueListener
    listener = QueueListener(queue, handler, respect_handler_level=True)
    listener.start()

    # Collect all results first, then deduplicate and write
    all_results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_batch, batch, queue): len(batch) for batch in batches}
            with tqdm(total=len(items), desc="Reducing rules") as pbar:
                for future in as_completed(futures):
                    all_results.extend(future.result())
                    pbar.update(futures[future])
    finally:
        listener.stop()

    stats = {"total": 0, "saved": 0, "duplicates": 0, "errors": 0}
    seen: set = set()
    print(f"Deduplicating and saving {len(all_results)} results...")
    with open(output_file, "w") as out_f:
        _write_results(all_results, out_f, stats, seen)

    log_structured(logger, "info", event="done", **stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce LHS and RHS of rewrite rules independently via greedy reduction")
    parser.add_argument("input", help="Input rules file (lhs->rhs format)")
    parser.add_argument("output", help="Output JSONL file for reduced rules")
    parser.add_argument("--log", default=None, help="Log file path (default: stdout)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    parser.add_argument("--chunk-size", type=int, default=50, help="Rules per parallel chunk (default: 50)")
    args = parser.parse_args()

    if args.workers > 1:
        configure_logging()
        stats = run_parallel(args.input, args.output, args.workers, args.log, args.chunk_size)
    else:
        configure_logging(args.log)
        stats = run_sequential(args.input, args.output)

    print(f"Saved {stats['saved']}/{stats['total']} rules "
          f"({stats['duplicates']} duplicates, {stats['errors']} errors) to {args.output}")


if __name__ == "__main__":
    main()
