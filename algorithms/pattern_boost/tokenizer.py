# Created by: Alex Myasnikov
# Date: May 14, 2026

"""FactoredWord tokenizer.

Lossless bidirectional FactoredWord <-> token id sequence.

Token stream per FactoredWord:
    [BOS] (factor)+ [EOS]
where (factor) = [REL_<idx>] (conj_letter)* [SEP]
      conj_letter = [CONJ_<+k>] / [CONJ_<-k>] for k in 1..G
                  | [CONJ_EMPTY]   (when the conjugator is empty)

BPE merges layered via HuggingFace `tokenizers`. Each base token is mapped
to a unique Unicode Private-Use-Area codepoint (U+E000+) so that stock BPE
operates on a "text" of codepoints. Trained tokenizer + schema is JSON-saved.
"""

import json
from pathlib import Path
from typing import Iterable, Optional

import tcgraph_ext as _ext
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

_pb = _ext.pattern_boost

PUA_START = 0xE000


def _conj_token_name(k: int) -> str:
    sign = '+' if k > 0 else '-'
    return f'[CONJ_{sign}{abs(k)}]'


class FactoredWordTokenizer:
    """Bidirectional FactoredWord ↔ token id encoder with BPE compression."""

    PAD = '[PAD]'
    BOS = '[BOS]'
    EOS = '[EOS]'
    SEP = '[SEP]'
    CONJ_EMPTY = '[CONJ_EMPTY]'

    def __init__(self, num_generators: int, num_relators: int):
        if num_generators < 1:
            raise ValueError(f'num_generators must be >= 1, got {num_generators}')
        if num_relators < 1:
            raise ValueError(f'num_relators must be >= 1, got {num_relators}')
        self.num_generators = num_generators
        self.num_relators = num_relators
        self._build_base_vocab()
        self._hf: Optional[Tokenizer] = None
        self._init_identity_tokenizer()

    def _build_base_vocab(self) -> None:
        base = [self.PAD, self.BOS, self.EOS, self.SEP, self.CONJ_EMPTY]
        for k in range(1, self.num_generators + 1):
            base.append(_conj_token_name(+k))
            base.append(_conj_token_name(-k))
        for i in range(1, self.num_relators + 1):
            base.append(f'[REL_{i}]')

        self.base_tokens: list[str] = base
        self.token_to_char: dict[str, str] = {
            t: chr(PUA_START + i) for i, t in enumerate(base)
        }
        self.char_to_token: dict[str, str] = {c: t for t, c in self.token_to_char.items()}

    def _init_identity_tokenizer(self) -> None:
        vocab = {self.token_to_char[t]: i for i, t in enumerate(self.base_tokens)}
        self._hf = Tokenizer(BPE(vocab=vocab, merges=[]))

    @property
    def vocab_size(self) -> int:
        return self._hf.get_vocab_size()

    def __len__(self) -> int:
        return self.vocab_size

    def _id_of(self, base_token: str) -> int:
        tid = self._hf.token_to_id(self.token_to_char[base_token])
        if tid is None:
            raise RuntimeError(f'base token {base_token} not present in tokenizer vocab')
        return tid

    @property
    def pad_token_id(self) -> int:
        return self._id_of(self.PAD)

    @property
    def bos_token_id(self) -> int:
        return self._id_of(self.BOS)

    @property
    def eos_token_id(self) -> int:
        return self._id_of(self.EOS)

    @property
    def sep_token_id(self) -> int:
        return self._id_of(self.SEP)

    def _fw_to_base_tokens(self, fw) -> list[str]:
        tokens = [self.BOS]
        for factor in fw.factors:
            idx = int(factor.relator_idx)
            if not (1 <= idx <= self.num_relators):
                raise ValueError(f'relator_idx {idx} out of range 1..{self.num_relators}')
            tokens.append(f'[REL_{idx}]')
            letters = list(factor.conjugator.get_letters())
            if not letters:
                tokens.append(self.CONJ_EMPTY)
            else:
                for k in letters:
                    if k == 0 or abs(k) > self.num_generators:
                        raise ValueError(
                            f'conjugator letter {k} out of range for G={self.num_generators}'
                        )
                    tokens.append(_conj_token_name(int(k)))
            tokens.append(self.SEP)
        tokens.append(self.EOS)
        return tokens

    def _base_tokens_to_codepoint_str(self, tokens: list[str]) -> str:
        try:
            return ''.join(self.token_to_char[t] for t in tokens)
        except KeyError as e:
            raise ValueError(f'unknown base token: {e}') from e

    def encode(self, fw, add_bos_eos: bool = True) -> list[int]:
        base = self._fw_to_base_tokens(fw)
        if not add_bos_eos:
            base = base[1:-1]
        s = self._base_tokens_to_codepoint_str(base)
        return self._hf.encode(s).ids

    def _ids_to_base_tokens(self, ids: Iterable[int]) -> list[str]:
        out: list[str] = []
        for i in ids:
            tok_str = self._hf.id_to_token(int(i))
            if tok_str is None:
                raise ValueError(f'token id {i} not in vocab')
            for c in tok_str:
                base = self.char_to_token.get(c)
                if base is None:
                    raise ValueError(f'codepoint U+{ord(c):04X} is not a base token')
                out.append(base)
        return out

    def decode(self, ids: Iterable[int]):
        base = self._ids_to_base_tokens(ids)
        return self._base_tokens_to_factored_word(base)

    def try_decode(self, ids: Iterable[int]):
        try:
            return self.decode(ids)
        except (ValueError, IndexError):
            return None

    def _base_tokens_to_factored_word(self, tokens: list[str]):
        i = 0
        n = len(tokens)
        while i < n and tokens[i] in (self.PAD, self.BOS):
            i += 1
        end = n
        for j in range(i, n):
            if tokens[j] in (self.EOS, self.PAD):
                end = j
                break

        fw = _pb.FactoredWord()
        factors = []
        cur_rel: Optional[int] = None
        cur_conj_letters: list[int] = []
        seen_empty = False

        for j in range(i, end):
            tok = tokens[j]
            if tok.startswith('[REL_'):
                if cur_rel is not None:
                    raise ValueError(f'expected [SEP] before {tok} at position {j}')
                cur_rel = int(tok[len('[REL_'):-1])
                if not (1 <= cur_rel <= self.num_relators):
                    raise ValueError(f'relator_idx {cur_rel} out of range')
                cur_conj_letters = []
                seen_empty = False
            elif tok == self.CONJ_EMPTY:
                if cur_rel is None:
                    raise ValueError(f'[CONJ_EMPTY] before [REL_] at position {j}')
                if cur_conj_letters or seen_empty:
                    raise ValueError(f'[CONJ_EMPTY] mixed with other content at position {j}')
                seen_empty = True
            elif tok.startswith('[CONJ_'):
                if cur_rel is None:
                    raise ValueError(f'{tok} before [REL_] at position {j}')
                if seen_empty:
                    raise ValueError(f'conjugator letter after [CONJ_EMPTY] at position {j}')
                k = int(tok[len('[CONJ_'):-1])
                cur_conj_letters.append(k)
            elif tok == self.SEP:
                if cur_rel is None:
                    raise ValueError(f'[SEP] without preceding [REL_] at position {j}')
                f = _pb.Factor()
                f.relator_idx = cur_rel
                f.conjugator = _ext.Word(cur_conj_letters)
                factors.append(f)
                cur_rel = None
                cur_conj_letters = []
                seen_empty = False
            else:
                raise ValueError(f'unexpected token {tok} at position {j}')

        if cur_rel is not None:
            raise ValueError('unterminated factor at end (missing [SEP])')
        if not factors:
            raise ValueError('decoded FactoredWord has zero factors')

        fw.factors = factors
        return fw

    def train_bpe(self, samples, vocab_size: int) -> None:
        if vocab_size < len(self.base_tokens):
            raise ValueError(
                f'vocab_size {vocab_size} smaller than base vocab {len(self.base_tokens)}'
            )

        corpus = [
            self._base_tokens_to_codepoint_str(self._fw_to_base_tokens(fw))
            for fw in samples
        ]

        tok = Tokenizer(BPE())
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=[self.token_to_char[t] for t in self.base_tokens],
            special_tokens=[],
            show_progress=False,
            min_frequency=1,
        )
        tok.train_from_iterator(corpus, trainer=trainer, length=len(corpus))
        self._hf = tok
        for t in self.base_tokens:
            if tok.token_to_id(self.token_to_char[t]) is None:
                raise RuntimeError(f'base token {t} missing from trained vocab')

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._hf.save(str(path / 'tokenizer.json'))
        meta = {'num_generators': self.num_generators, 'num_relators': self.num_relators}
        (path / 'schema.json').write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path) -> 'FactoredWordTokenizer':
        path = Path(path)
        meta = json.loads((path / 'schema.json').read_text())
        obj = cls(num_generators=meta['num_generators'], num_relators=meta['num_relators'])
        obj._hf = Tokenizer.from_file(str(path / 'tokenizer.json'))
        for t in obj.base_tokens:
            if obj._hf.token_to_id(obj.token_to_char[t]) is None:
                raise RuntimeError(f'base token {t} missing in loaded tokenizer')
        return obj
