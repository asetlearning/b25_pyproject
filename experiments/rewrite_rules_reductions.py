"""
Read rewrite rules and try to reduce by removing 2.5 powers

Created by: Alex Myasnikov
Date: March 18, 2026
"""

import argparse
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from algorithms.greedy_reduction import greedy_reduce_cyclic_exhaustive
from algorithms.logger import configure_logging, configure_logging_queue, get_logger, log_structured, \
    JsonFormatter
from algorithms.rewrite_rules import iter_rules_file
from algorithms.word_problem import word_from_str, greedy_reduce_cyclic_exhaustive2_5, greedy_reduce2_5, \
    word_to_str

logger = get_logger(__name__)


def _reduce_one(idx: int, rule_str: str, word: str) -> dict:
    """Reduce a single rule word. Module-level for picklability."""
    name = f"rule_{idx}"
    original_len = len(word)
    log_structured(logger, "info", event="processing_rule", rule=rule_str, word_len=original_len)
    try:
        #reduced = greedy_reduce_cyclic_exhaustive(word, name, show_progress=False)
        reduced = greedy_reduce_cyclic_exhaustive2_5(word)
        # reduced = greedy_reduce2_5(word)

        reduction = original_len - len(reduced)
        log_structured(
            logger, "info",
            event="rule_result",
            rule=rule_str,
            original_len=original_len,
            reduced_len=len(reduced),
            reduction=reduction,
        )
        return {
            "rule": rule_str,
            "original_word": word,
            "original_len": original_len,
            "reduced_word": reduced,
            "reduced_len": len(reduced),
            "reduction": reduction,
            "error": None,
        }
    except Exception as e:
        log_structured(logger, "error", event="reduction_error", rule=rule_str, error=str(e))
        return {
            "rule": rule_str,
            "original_word": word,
            "original_len": original_len,
            "reduced_word": word,
            "reduced_len": original_len,
            "reduction": 0,
            "error": str(e),
        }


def _process_batch(batch: list, queue) -> list:
    """Process a batch of (idx, rule_str, word) triples in a worker process."""
    configure_logging_queue(queue)
    return [_reduce_one(idx, rule_str, word) for idx, rule_str, word in batch]


def _write_results(results: list, out_f, stats: dict) -> None:
    for r in results:
        if r["error"] is not None:
            continue
        if r["reduction"] > 0:
            out_f.write(json.dumps({k: v for k, v in r.items() if k != "error"}) + "\n")
            out_f.flush()
            stats["saved"] += 1
        stats["total"] += 1


def run_sequential(input_file: str, output_file: str) -> dict:
    stats = {"total": 0, "saved": 0}
    items = [(idx, rule) for idx, rule in enumerate(iter_rules_file(input_file))]

    with open(output_file, "w") as out_f:
        for idx, rule in tqdm(items, desc="Reducing rules"):
            word = rule.to_word()
            result = _reduce_one(idx, str(rule), word)
            _write_results([result], out_f, stats)

    log_structured(logger, "info", event="done", **stats)
    return stats


def run_parallel(
    input_file: str,
    output_file: str,
    workers: int,
    log_file: str | None = None,
    chunk_size: int = 50,
) -> dict:
    items = [(idx, str(rule), rule.to_word()) for idx, rule in enumerate(iter_rules_file(input_file))]
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

    stats = {"total": 0, "saved": 0}
    try:
        with open(output_file, "w") as out_f:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_process_batch, batch, queue): len(batch) for batch in batches}
                with tqdm(total=len(items), desc="Reducing rules") as pbar:
                    for future in as_completed(futures):
                        _write_results(future.result(), out_f, stats)
                        pbar.update(futures[future])
    finally:
        listener.stop()

    log_structured(logger, "info", event="done", **stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce rewrite rules via greedy cyclic exhaustive reduction")
    parser.add_argument("input", help="Input rules file (lhs->rhs format)")
    parser.add_argument("output", help="Output JSONL file for reduced words")
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

    print(f"Saved {stats['saved']}/{stats['total']} rules to {args.output}")


if __name__ == "__main__":
    main()
    exit(0)
    ws = ['AAbaaabAbaaabAAbaBAbaabaabABAABaaBab',
          'aaabaaabAbaaabaaabaBAbaabaabABAABaaBab',
          'aaabaaaaaabABABAABabAAAbaabaBAABABAbaaa',
          'BAbaabaabABAABaaBabABAAABaBAAABaBaaBAba',
          'AAABAABAABBABABAAAAAABABAAABAAAAAAABAAAA',
          'abaBAABAABabaBAABAABabaaabABAAABaBAAABAbaa',
          'baaaabaaaabAABAAAAAABABABAAABABABAAAAAABAA',
          'abaBAABAABabaBAABaaaBabaaabABAAABaBAAABAbaa',
          'aaBAbaabaabbaBBAAABAAABAbaabaabbaBAAbaBBAAAB',
          'BAbaabaabbaBBAAABAAABAbaabaabbaBAAbaBBAAABaa',
          'AABBBABaabABBAABAABabaaabaaabbABBAABAABabAAb',
          'ABabaaabaaabbABBAABAABabAAbAABBBABaabABBAABA',
          'bAABBBABaabABBAABAABabaaabaaabbABBAABAABabAA',
          'AAbAABBBABaabABBAABAABabaaabaaabbABBAABAABab',
          'ABAbaabaabbaBAAbaBBAAABaaBAbaabaabbaBBAAABAA',
          'ABBBABaabABBAABAABabaaabaaabbABBAABAABabAAbA',
          'AbaabaabbaBAAbaBBAAABaaBAbaabaabbaBBAAABAAAB',
          'baaaBBBABaabABBAABAABabaaabaaabbABBAABAABabAA',
          'aabAABBBABaabABBAABAABabaaabaaabbABBAABAABaba',
          'aaaaabaabaabbaabAAAABAAAAAABABABabaabaaaabaabaa',
          'AABABABabaabaaaabaabaaaaaaabaabaabbaabAAAABAAAA',
          'BAbaabaabABAAABabAbABaBBaBAbAbaBAAABAbaabaabABaa',
          'AABABBABABababaaababaaaaaabababbabaababAAAABAAAA',
          'BaBAAABAbaaabAbaaabAAbaaabaBAABabaBAABAABabaaabABaa',
          'aBaBBBABaabABBAABAABabaaabaaabbABBAABAABabaBAABAABaa',
          'AAbbaaabaaBAbaabaabbaBBAAABAAABAbaabaabbaBAAbAbbaaabb',
          'aabAABAABaBAABaBBAABAABBAABAABaabaabAbaabAbbaabaabbaab',
          'abAABAABaBAABaBBAABAABBAABAABaabaabAbaabAbbaabaabbaaba',
          'ABAAAbABBAABabaBAABAABabaaabABAAABaBAAABABBBABaabABaab',
          'AABAABaBAABaBBAABAABBAABAABaabaabAbaabAbbaabaabbaabaab',
          'baBAABAABabaBAAbbbAbAAABAbAAABAbaaBABBBABAAABaBAAABAbAA',
          'aBAbaaabaBAABabaBAABAABabaaabABAAABaBAAABBAAABaBAAABaBa',
          'ABaaBAAABaBAAABaabbbabAABabbaabaabABAAABAAABBaBBBaabABAA',
          'AAABaaBAAABaBAAABaabbbabAABabbaabaabABAAABAAABBaBBBaabAB',
          'AAAbABAAABaaBBaBaBBaabABBAABAABabaaabaaabbABBABAABABAABab',
          'AbaaBBBAABBBABaabABBAABAABabaaabaaabbABBAABAABabbaabaabAA',
          'BAAABAAbaaabaaBABaabaaababaaabaaabAABAAABBAbaabaabABBAAAB',
          'ABabaaabaaabbABBAABAABabbaabaabAAAbAAABBBAABBBABaabABBAABA',
          'baaBAAABAAABABAbaabaabbaBBAAABAAABAbaabaabbaBAAbAbbaaabbAA',
          'abaBAABAABabABAABAAbaabaabAABAABaBAABAbaaabABAAABaBAAABAbaa',
          'aBAABAABBAbaaBABBBABAAABaBAAABaBaaBAbaaabaBAAbbbAbbaaabaaab',
          'abaaabaaabbABBAABAABababaaabaaabAAbaaabbaaabbABaabABBAABAAB',
          'baBAbaabaabABAABabAbaabaBAAbaBaaabaBAAbaBAABBAAbaaabAbaaabAA',
          'baaBAAABABAAABABAABAAbaBBAAABAAABAbaabaabbaBAAbAbaaabbaaabAA',
          'AAbaaBAAABABAAABABAABAAbaBBAAABAAABAbaabaabbaBAAbAbaaabbaaab',
          'aabAABAABaBAABABaBaBBaabABBAABAABabaaabaaabbABaababbaabaabbaab',
          'baaabABAAABaBabaabaaBAABAABaabaabAbaabAbAABAAABAbaaabaBAABAABa',
          'aBAABAABabaaabABAAABaBabaabaaBAABAABaabaabAbaabAbAABAAABAbaaab',
          'ABabaaabaaabbABBAABAABabbbaabbaabbAAABBBAAABBAABBBABaabABBAABA',
          'AABAABabaaabABAAABaBabaabaaBAABAABaabaabAbaabAbAABAAABAbaaabaB',
          'BAAABBAABaBAABaBBAABAABBAABAABaabaabAbaabAbbaabaabbaabABAAABaa',
          'aBAABAABBAbaaBABBBAbaaabaBAAABAbaabaabABAbaabaabbababbbaaabaaab',
          'AAABAABBAbaaBABBBaaabaaabaBAAAAAAABAAbbbaaaabbaaabaaabAAAABAAAA',
          'aBAABAABBAbaaBABBBABAAABaBAAABAbAAbaBAABAABabaBAAbbbAbbaaabaaab',
          'aaBAABabaaabABAAABaBabaabaaBAABAABaabaabAbaabAbAABAAABAbaaabaBa',
          'baaabAbaaababbbabAABabbaabaabABAAABAAABBaBBBaabABAbaabaabABaaBa',
          'AAABAABAABBAAABABAAABABAAbabaababaaaaaabababbAABaabaaabAAAABAAAA',
          'aaBAABAABBAABaBBBABaabABBAABAABabaaabaaabbABBAABAABabaBAABaBAABa',
          'AAAABAAAAbababaababbaaabaaabAAAABAAAAAAABAABAABabaaaabaBABBABBAA',
          'AbaaBaBAABaBAABABBBABaabABBAABAABabaaabaaabbABBAABAABabbAbAbabAb',
          'aabAABAABaBAABAABBBaBAAbaBBAAABAAABAbaabaabbaBAAbaaabbaabaabbaab',
          'AbaabaabbaBAAbABabAABabbaabaabABAAABAAABBabaaaBAbaBAAbaBBAAABAAAB',
          'aBAABAABBAbaaBAbaBaabABBAABAABabaaabaaabbABaabABabAAABAbbaaabaaab',
          'BaBAbAbabbbabAABabbaabaabABAAABAAABBaBBBaabaaaBAAABaBAAABaBabABaB',
          'AbaabaabbbAbabAbAbabAbAbbAbaaabAbABabAbbAbaBaBAbabaBAAbaBBAAABAAAB',
          'AAbaaBAAABABAAABaBABABBAAbaBBAAABAAABAbaabaabbaBAAbAbbaaabaaabbaaab',
          'abaBAABAABabbaabAbaabAAABAAbaabbAAABBBaabaaaBBabaaabABAAABaBAAABAbaa',
          'ABabaaabaaabbABaababbaabaabbaabABaaBAAABAAABBAABaBAABABaBaBBaabABBAABA',
          'bbAABAAABAAbaaabaaababbAABaabbbAAABBAAABAAbaaBAAbabAABABAAABAAAbaaabAA',
          'BAAABBAAbaaabaaBABaabaaababaaabaaabAABAAABBAbaabaabABBAAABBABAAABAAABaa',
          'abAAbaaabAbaaBBBAABBBABaabABBAABAABabaaabaaabbABBAABAABabAABAABAbaaabaa',
          'aaabbaaaBAAABBAAABBAbAbaabAAABAAbaabbAAABBBAAABBAAABBBAAABBababAABaabAAb',
          'AABAbaabaabbaBAAbAbbaaabaabbbaaBAAABBAAABaaBBAAABAAABBABAbaabaabbaBBAAABA',
          'aabaabAbaabAbaBaBBAABabAABabbaabaabABAAABAAABBabaaaBaBBAbabAbaabaaBAABAAB',
          'AAABABAAABAbbaaabbaaabbAAAABAAAAAAABAABAABBAAABAAABAABAAAABAAAABAABABABAAA',
          'aabaabAbaabAbaBaBBAAABAbAAABAbbaaabaaabaBAABAABBAbaaBABBAbabAbaabaaBAABAAB',
          'aaBAABAABBabaaabABAAABaBabaabaaBAABAABaabaabAbaabAbAABAAABAbaaabaBAABaBAABa',
          'baaabaaabbABaabbabAABaabAABAAABAAbaaabaaabAABAAABAAbaaBAAABAABaabABBAABAABa',
          'abaBAABAABababbbAbAAABAbbaaabaaabaBAABAABBABBBAAbaaabAbaaabABAAABaBAAABAbaa',
          'aaabaabaabbaabbbaaaabbaaaabbAABAAAAAABABABBAAAABAAAABBABABabaabaaaabaabaaaa',
          'bbaaabaaabbabaaabaaBABaabaaababaaabaaabAABAAABBAbaabaabABAABBBAAbaaabbaaabAA',
          'AABAbaabaabbaBAAbAbbaaabaabbbaabaaabbAABBBAAbaaBBAAABAAABBABAbaabaabbaBBAAABA',
          'AAAABAAAAbababaababbbaaabbaaabbAAAABAAAAAAABAABAABBAAABAAABaabaaaabaBABBABBAA',
          'AABABABBABABAAABABABabaabaaaabaabaaaaaaabaabaabbaabbaaaabaaaabbaabaabAAAABAAAA',
          'BaBAAABaBBAABaBAABaBBAABAABBAABAABaabaabAbaabAbbaabaabbaabAABAAABAbaaabaaabABaa',
          'aBAABAABBAbaaBABBBAAbabaBAAbaBBAAABAAABAbaabaabbbAbAAABAAbaabaabbaababbaaabaaab',
          'baaBAAABABAbaabaabABaBAAABaBabaabaaBAABAABaabaabAbaabAbAABAAABAbbaaabaaabbaaabAA',
          'AbaabaabbaBAAbAbaaabAABAABabAABabbaabaabABAAABAAABBabaaaBaBBAAbabaBAAbaBBAAABAAAB',
          'aaabaabaabbaabbaaaabaaaabbaaaabaaaabAABAAAAAABABABBAAABAAABBABABabaabaaaabaabaaaa',
          'AbaabaabbaBAAbAbaaabAABAAABAbAAABAbbaaabaaabaBAABAABBAbaaBABBAAbabaBAAbaBBAAABAAAB',
          'AbaabaabbaBAAbAbaaababbbAbAAABAbbaaabaaabaBAABAABBAbaaBAAABAABBABABBAAbaBBAAABAAAB',
          'aBAABAABBAbaaBABBBABAAABaBAAABaBabaabaaBAABAABaabaabAbaabAbabaaabaBAAbbbAbbaaabaaab',
          'aBAABAABBAbaaBABBBAAbabaBaaabAbaaabAAbAAABABAbbbaaabbAbAAABAAbaabaabbaababbaaabaaab',
          'AAABABAAABAbababbbaaabbaaabbAAAABAAAAAAABAABAABBAAABAAABAABAAAABAAAABBAABAABBABABAAA',
          'ababAABaabaaabbAbAbabbbabAABabbaabaabABAAABAAABBabaaaBAABAAABabaabaaBAAABABAAABABAAb',
          'aabaabAbaabAbaBBaBBaabbbabAABabbaabaabABAAABAAABBabaaaBAABAAABAAAbAbbabAbaabaaBAABAAB',
          'aaabaaabAABaabaaabbAABAABabAABabbaabaabABAAABAAABBabaabAbaaBAABAABBBAAABAAABBAAABAAAB',
          'baaabaaababbaaabaaabbabaaabaaBABaabaaababaaabaaabAABAAABBAbaabaabABAABBBAABAAABBAAABAA',
          'aabaabAbaabAABAABABaBBAAABAbAAABAbbaaabaaabaBAABAABBAbaaBABBAbabaababaabAbaabaaBAABAAB',
          'AAbaaBABABaabaabAbaabaaaBBBABaabABBAABAABabaaabaaabbABBAABAABabbAbAbbaaabaaabbaaabaaab',
          'ABaaBAAABaBAAbaaBAABBAABBabaaabaabAAABAbbaaabaaabaBAABAABBAbaaBABaabaabAABAABaBABAAABAA',
          'aBAABAABBAbaaBABBBABAAABaBAAABaBabaaabaBAAABAAABabaaabAbaaabAbaabaaabaBAAbbbAbbaaabaaab',
          'AAABAABAABBAAABAAABAABAAAABAAAABBAABAABAbabaababaaaaaabababbabbababbbaaabbaaabbAAAABAAAA',
          'BBAAABAABBBAABAbaabaabABBAAABBABAAABAAABaaBAAABBAAbaaabaaBABaabaaababbaaabaaabbAABaabbbaa',
          'AAABABABBAABAABAAAABABABBABABAAABABABAAAAAABAAbaaaabaaaabbaabaabbaaaabaaaabbaaabaaabAAABAAA',
          'baBAABAABaababaababAbaabAABBBABaabABBAABAABabaaabaaabbABBAABAABabbAbAbbaabaabbAbAbbaaabaaabAA',
          'AAABAABAABBAAABAAABAABAAAABAAAABBAABAABAbabaaababaaaaaabababaaaababbababbbaaabbaaabbAAAABAAAA',
          'ABAbaabaabABBAAABBABAAABAAABaaBAAABBAAbaaabaaBABaabaaababbaaabaaabbabaaabaaabAABAAABBAAABAABBBA',
          'AAbaBAABAABaababaababaaBAABabaaabABAAABaBabaabaaBAABAABaabaabAbaabAbAABAAABAbaaabbAbAbbaabaabbAbAbbaaabaaab'
    ]

#    ws = ["AAbaaabAbaaabAAbaBAbaabaabABAABaaBab"]
    print(word_from_str(ws[0]))

    print("Compute original ... ")
    reduced = [greedy_reduce_cyclic_exhaustive(w, "test", show_progress=False) for w in ws]
    print("Compute CPP ... ")
    reduced2 = [greedy_reduce_cyclic_exhaustive2_5(w) for w in ws]
    print("Compare ... ")
    for i in range(len(reduced)):
        if reduced[i] != word_to_str(reduced2[i]):
            print(word_from_str(reduced[0]))
            print(reduced2[0])


    # reduced_list = []
    # with open("/Users/alex/Downloads/trivial_no_powers.json") as f:
    #     for w in f.readlines():
    #         reduced = greedy_reduce_cyclic_exhaustive(w.strip(), "test", show_progress=False)
    #         if len(reduced) > 0:
    #             reduced_list.append(reduced)
    #
    # print(len(reduced_list), len(set(reduced_list)), set(reduced_list))
    #
    # # reduced = [greedy_reduce_cyclic_exhaustive(w.strip(), "test", show_progress=False) for w in reduced_list]
    # print([r for r in reduced if "aaa" in r or "bbb" in r or "AAA" in r or "BBB" in r])
    #
    # # reduced_rules_set_json = {}
    # # for i, r in enumerate(set(reduced_list)):
    # #     reduced_rules_set_json[f"rukle{i}"] = {
    # #         "version": "0",
    # #         "word": r
    # #     }
    # # with open("reduced_rules.json", "w") as f:
    # #     json.dump(reduced_rules_set_json, f)
    # print("\n".join(sorted(set(reduced_list), key=len)))