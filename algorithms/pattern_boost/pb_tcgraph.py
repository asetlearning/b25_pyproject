# Created by: Alex Myasnikov
# Date: April 24, 2026

import tcgraph_ext as _ext
from algorithms.word_conversion import string_to_word

_pb = _ext.pattern_boost

# Re-export types
Factor = _pb.Factor
FactoredWord = _pb.FactoredWord
RelatorMap = _pb.RelatorMap
SampleConfig = _pb.SampleConfig
ConjugatorLenSampler = _pb.ConjugatorLenSampler
BigramFrequencies = _pb.BigramFrequencies
WordScorer = _pb.WordScorer
HellingerScorer = _pb.HellingerScorer
HammingScorer = _pb.HammingScorer
EditScorer = _pb.EditScorer
LocalSearchConfig = _pb.LocalSearchConfig
LocalSearcher = _pb.LocalSearcher

# Re-export free functions
build_standard_relators = _pb.build_standard_relators
uniform_len = _pb.uniform_len
zero_heavy_len = _pb.zero_heavy_len
geometric_len = _pb.geometric_len
constant_len = _pb.constant_len
generate_word = _pb.generate_word
generate_words = _pb.generate_words
to_json = _pb.to_json
from_json = _pb.from_json
hellinger_distance = _pb.hellinger_distance
hamming_distance = _pb.hamming_distance
edit_distance = _pb.edit_distance
load_challenges = _pb.load_challenges


def score_word(word, scorer):
    """Score a word (str or Word) using a pre-built WordScorer."""
    if isinstance(word, str):
        word = _ext.Word(string_to_word(word))
    return scorer.score_word(word)
