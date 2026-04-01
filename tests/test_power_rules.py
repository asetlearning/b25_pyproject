# Created by: Alex Myasnikov
# Date: February 26, 2026

import pytest

from algorithms.power_rules import create_power_rewrite_rule
from algorithms.word_conversion import word_inverse


class TestWordInverse:
    def test_known_value(self):
        assert word_inverse("ab") == "BA"

    def test_known_value_mixed(self):
        assert word_inverse("abAB") == "baBA"

    def test_roundtrip(self):
        for w in ["ab", "ababab", "AbBa", "aAbB"]:
            assert word_inverse(word_inverse(w)) == w

    def test_empty(self):
        assert word_inverse("") == ""

    def test_single_lower(self):
        assert word_inverse("a") == "A"

    def test_single_upper(self):
        assert word_inverse("B") == "b"


class TestCreatePowerRewriteRule:
    def test_integer_power_p3(self):
        # W="ab", p=3: lhs=W^3="ababab", W^q=W^2="abab", rhs=inverse("abab")="BABA"
        r = create_power_rewrite_rule("ab", 3)
        assert r.lhs == "ababab"
        assert r.rhs == "BABA"

    def test_fractional_power_p3_5(self):
        # W="abab" (n=4), p=3.5: lhs_len=14, W^5="abababababababababab" (20 chars)
        # w_p = first 14 chars, w_q = last 6 chars
        word = "abab"
        r = create_power_rewrite_rule(word, 3.5)
        w5 = word * 5
        assert r.lhs == w5[:14]
        assert r.rhs == word_inverse(w5[14:])

    def test_p_too_low_strict(self):
        with pytest.raises(ValueError, match="2.5 < p < 5"):
            create_power_rewrite_rule("ab", 2.0)

    def test_p_too_high_raises(self):
        with pytest.raises(ValueError, match="2.5 < p < 5"):
            create_power_rewrite_rule("ab", 5)

    def test_p_too_high_strict(self):
        with pytest.raises(ValueError, match="2.5 < p < 5"):
            create_power_rewrite_rule("ab", 6)

    def test_non_integer_product_raises(self):
        # p=3.5, n=2: 3.5*2=7.0 — this is actually integer, use n=3
        # p=3.5, n=3: 3.5*3=10.5 — not integer
        with pytest.raises(ValueError, match="must be an integer"):
            create_power_rewrite_rule("abc", 3.5)

    def test_lhs_longer_than_rhs(self):
        # lhs = W^p (p>2.5), rhs = inverse(W^q) where q=5-p < 2.5
        # so len(lhs) = p*n > 2.5n and len(rhs) = (5-p)*n < 2.5n => lhs longer
        r = create_power_rewrite_rule("ab", 3)
        assert len(r.lhs) > len(r.rhs)

    def test_repr(self):
        r = create_power_rewrite_rule("ab", 3)
        assert repr(r) == "ababab->BABA"

    def test_fp_noise_exponent(self):
        # sub_len=5098, match_len=2550 → p*5098 should be 22942 despite FP noise
        sub_len = 5098
        match_len = 2550
        p = 4 + match_len / sub_len
        word = "a" * sub_len
        r = create_power_rewrite_rule(word, p)
        assert len(r.lhs) == 22942
        assert len(r.rhs) == 5 * sub_len - 22942
