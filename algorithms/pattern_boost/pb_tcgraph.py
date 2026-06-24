# Created by: Alex Myasnikov
# Date: April 24, 2026

import tcgraph_ext as _ext
from algorithms.pattern_boost.relators import build_hybrid_relators
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
ScorerConfig = _pb.ScorerConfig
HellingerScorer = _pb.HellingerScorer
HammingScorer = _pb.HammingScorer
EditScorer = _pb.EditScorer
WordProductLengthScorer = _pb.WordProductLengthScorer
AbelianizedProductLengthScorer = _pb.AbelianizedProductLengthScorer
HighPowerCountScorer = _pb.HighPowerCountScorer
LengthRangeScorer = _pb.LengthRangeScorer
HybridScorer = _pb.HybridScorer
ConjMoveStrategy = _pb.ConjMoveStrategy
LocalSearchConfig = _pb.LocalSearchConfig
LocalSearcher = _pb.LocalSearcher
BPEModel = _pb.BPEModel
BPETokenizer = _pb.BPETokenizer
WordCompressor = _pb.WordCompressor

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
make_hybrid_scorer = _pb.make_hybrid_scorer
length_range_score = _pb.length_range_score


def score_word(word, scorer):
    """Score a word (str or Word) using a pre-built WordScorer."""
    if isinstance(word, str):
        word = _ext.Word(string_to_word(word))
    return scorer.score_word(word)


def bpe_model_from_compressor(py_tok) -> BPEModel:
    """Convert a BPETokenizerWordCompression instance to a C++ BPEModel.

    bpe_merges insertion order encodes merge priority (matches _build_merge_priority).
    """
    merges = [(p[0], p[1], v) for p, v in py_tok.bpe_merges.items()]
    return BPEModel(vocab=py_tok.vocab, merges=merges, eow=py_tok.eow)
