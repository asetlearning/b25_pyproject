"""
Tests for rewrite rules system.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

import pytest
from algorithms.rewrite_rules import RewriteRule, apply_rule, apply_rules


def test_single_replacement():
    """Test single occurrence replacement"""
    rule = RewriteRule("aa", "b")
    word = "xaay"
    result, changed = apply_rule(word, rule)
    assert result == "xby"
    assert changed


def test_multiple_occurrences():
    """Test multiple occurrences replaced"""
    rule = RewriteRule("aa", "b")
    word = "aaaa"
    result, changed = apply_rule(word, rule)
    assert result == "bb"
    assert changed


def test_no_match():
    """Test no replacement when lhs not in word"""
    rule = RewriteRule("aa", "b")
    word = "xyz"
    result, changed = apply_rule(word, rule)
    assert result == "xyz"
    assert not changed


def test_deletion_rule():
    """Test rule with empty rhs (deletion)"""
    rule = RewriteRule("aaaaa", "")
    word = "xaaaaay"
    result, changed = apply_rule(word, rule)
    assert result == "xy"
    assert changed


def test_multiple_deletions():
    """Test multiple deletions in same word"""
    rule = RewriteRule("aa", "")
    word = "aaaaaa"
    result, changed = apply_rule(word, rule)
    assert result == ""
    assert changed


def test_apply_rules_sequence():
    """Test applying multiple rules in sequence"""
    rules = [
        RewriteRule("aa", "b"),
        RewriteRule("bb", "c"),
    ]
    word = "aaaa"
    result = apply_rules(word, rules)
    assert result == "c"


def test_apply_rules_order_matters():
    """Test that rule order affects result"""
    rules = [
        RewriteRule("ab", "x"),
        RewriteRule("ba", "y"),
    ]
    word = "aba"
    result = apply_rules(word, rules)
    # First rule: aba -> xa
    # Second rule: xa -> xa (no match)
    assert result == "xa"


def test_overlapping_patterns():
    """Test overlapping pattern replacement"""
    rule = RewriteRule("aba", "x")
    word = "ababa"
    result, changed = apply_rule(word, rule)
    # Should replace leftmost first: ababa -> xba
    # Then continue: xba -> xba (no more match)
    assert result == "xba"
    assert changed


def test_expansion_rule():
    """Test rule that increases length"""
    rule = RewriteRule("a", "bbb")
    word = "aa"
    result, changed = apply_rule(word, rule)
    assert result == "bbbbbb"
    assert changed
