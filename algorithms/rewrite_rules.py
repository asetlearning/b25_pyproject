"""
Rewrite rule system for Knuth-Bendix completion rules.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

from typing import List, Tuple, Optional
from tqdm import tqdm

from algorithms.logger import get_logger
_logger = get_logger(__name__)

class RewriteRule:
    """Single rewrite rule: lhs -> rhs"""

    def __init__(self, lhs: str, rhs: str):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"{self.lhs}->{self.rhs}"


def parse_rules_file(filepath: str) -> List[RewriteRule]:
    """
    Parse rewrite rules from file.

    Format:
    - Lines starting with # are comments
    - Rules: lhs->rhs
    - Empty rhs allowed (deletion)
    """
    rules = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '->' not in line:
                continue

            lhs, rhs = line.split('->', 1)
            if len(lhs) < len(rhs):
                _logger.error(f"Invalid rule: {line}")
                continue
            rules.append(RewriteRule(lhs, rhs))

    return rules


def apply_rule(word: str, rule: RewriteRule) -> Tuple[str, bool]:
    """
    Apply single rule to word repeatedly until lhs no longer appears.

    Returns:
        (new_word, changed): Changed is True if any rewrite occurred
    """
    changed = False
    while rule.lhs in word:
        word = word.replace(rule.lhs, rule.rhs, 1)
        changed = True
    return word, changed


def apply_rules(word: str, rules: List[RewriteRule], progress_desc: Optional[str] = None) -> str:
    """
    Apply all rules from set to word.
    Each rule applied exhaustively in order.

    Args:
        word: The word to rewrite
        rules: List of rewrite rules to apply
        progress_desc: Optional description for progress bar
    """
    rule_iter = tqdm(rules, desc=progress_desc, leave=False) if progress_desc else rules

    for rule in rule_iter:
        new_word, _ = apply_rule(word, rule)
        if len(new_word) < len(word):
            _logger.info(f"Word shortened {len(word)} -> {len(new_word)} by rule {rule}")
        word = new_word

    return word


def apply_rules_exhaustive(word: str, rules: List[RewriteRule], max_iterations: int = 1000) -> str:
    """
    Apply rules repeatedly until no more changes or max iterations reached.
    """
    for _ in range(max_iterations):
        changed = False
        for rule in rules:
            new_word, rule_changed = apply_rule(word, rule)
            if rule_changed:
                word = new_word
                changed = True

        if not changed:
            break

    return word
