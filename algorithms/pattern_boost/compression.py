# Created by: Alex Myasnikov
# Date: May 29, 2026

"""WordCompressor: maps BPE tokens to PatternBoost generators and back."""

import tcgraph_ext as _ext

from algorithms.word_compression import BPETokenizerWordCompression
from algorithms.word_conversion import word_inverse


class WordCompressor:
    """Bridges BPETokenizerWordCompression and PatternBoost generator space.

    Each non-eow BPE token gets a 1-to-1 mapping to a generator number 1..N,
    where N = vocab_size - 1 (eow excluded). Negative generator -k decodes to
    the group-theoretic inverse of the token string for generator +k.
    """

    def __init__(self, path: str):
        self._bpe = BPETokenizerWordCompression.from_file(path)
        eow = self._bpe.eow
        eow_id = self._bpe.inverse_vocab[eow]

        non_eow_ids = sorted(tid for tid in self._bpe.vocab if tid != eow_id)
        self._token_to_gen: dict[int, int] = {tid: i + 1 for i, tid in enumerate(non_eow_ids)}
        self._gen_to_token: dict[int, int] = {g: tid for tid, g in self._token_to_gen.items()}
        self.num_generators: int = len(non_eow_ids)

    def encode_str(self, word_str: str) -> _ext.Word:
        """BPE-encode word_str (no eow) → _ext.Word in compressed generator space."""
        token_ids = self._bpe.encode(word_str)
        gen_nums = [self._token_to_gen[tid] for tid in token_ids]
        return _ext.Word(gen_nums)

    def decode_word(self, word: _ext.Word) -> str:
        """Decode a compressed _ext.Word back to original {a,b,A,B} string.

        Positive gen k → BPE token string; negative gen -k → its group inverse.
        No free-reduction is applied (consistent with non-compressed path).
        """
        parts = []
        for k in word.get_letters():
            tok_str = self._bpe.vocab[self._gen_to_token[abs(k)]]
            parts.append(tok_str if k > 0 else word_inverse(tok_str))
        return "".join(parts)

    def compressed_repr(self, word: _ext.Word) -> str:
        """Human-readable compressed form: [ab][BA][a] (BPE token strings in brackets)."""
        parts = []
        for k in word.get_letters():
            tok_str = self._bpe.vocab[self._gen_to_token[abs(k)]]
            s = tok_str if k > 0 else word_inverse(tok_str)
            parts.append(f"[{s}]")
        return "".join(parts)
