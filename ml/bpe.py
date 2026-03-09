from collections import Counter
import json


class BPETokenizerWords:
    def __init__(self, alphabet: list[str], eow: str = "#"):
        self.alphabet = alphabet
        self.eow = eow
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}

    def train(self, words: list[str] | str, vocab_size: int):
        """
        Train the BPE tokenizer from scratch.

        Args:
            words (list[str] | str): List of words or pre-joined training text.
            vocab_size (int): The desired vocabulary size.
        """

        if isinstance(words, list):
            processed_text = self.eow.join(words) + self.eow
        else:
            processed_text = words

        # Initialize vocab with alphabet plus eow characters
        unique_chars = self.alphabet + [self.eow]

        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Tokenize the processed_text into token IDs
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        eow_id = self.inverse_vocab[self.eow]

        # BPE steps 1-3: Repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most", exclude_id=eow_id)
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

        self._build_merge_priority()

    def encode(self, words, skip_full_words=False):
        """
        Encode the input text into a list of token IDs, with tiktoken-style handling of special tokens.

        Args:
            text (str): The input text to encode.
            allowed_special (set or None): Special tokens to allow passthrough. If None, special handling is disabled.

        Returns:
            List of token IDs.
        """

        tokens = []
        current_word = ""
        for w in words:
            if w == self.eow:
                if current_word:
                    tokens.append(current_word)
                    tokens.append(self.eow)
                    current_word = ""
            else:
                current_word += w

        if current_word:
            tokens.append(current_word)

        token_ids = []
        # Map tokens -> ids (BPE if needed)
        for tok in tokens:
            if not skip_full_words and tok in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[tok])
            else:
                token_ids.extend(self.tokenize_with_bpe(tok))

        return token_ids

    def _build_merge_priority(self):
        """Build priority lookup from bpe_merges insertion order."""
        self.merge_priority = {pair: rank for rank, pair in enumerate(self.bpe_merges)}

    def tokenize_with_bpe(self, token):
        """
        Tokenize a single token using BPE merges with correct priority ordering.

        Args:
            token (str): The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        while len(token_ids) > 1:
            # Find the highest-priority (lowest rank) mergeable pair
            best_pair = None
            best_rank = float('inf')
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                rank = self.merge_priority.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_id = self.bpe_merges[best_pair]
            new_tokenids = self.replace_pair(token_ids, best_pair, new_id)
            # TODO: tmp hack
            # if len(new_tokenids) == 1:
            #     break
            token_ids = new_tokenids

        return token_ids


    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        out = []
        for tid in token_ids:
            if tid not in self.vocab:
                raise ValueError(f"Token ID {tid} not found in vocab.")
            tok = self.vocab[tid]
            out.append(tok)

        return "".join(out)

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Save the vocabulary and BPE merges to JSON files.

        Args:
            vocab_path (str): Path to save the vocabulary.
            bpe_merges_path (str): Path to save the BPE merges.
        """
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        # Save BPE merges as a list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Load the vocabulary and BPE merges from JSON files.

        Args:
            vocab_path (str): Path to the vocabulary file.
            bpe_merges_path (str): Path to the BPE merges file.
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id
        self._build_merge_priority()

    def save(self, path: str):
        data = {
            "alphabet": self.alphabet,
            "eow": self.eow,
            "vocab": self.vocab,
            "bpe_merges": [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.bpe_merges.items()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_file(cls, path: str) -> "BPETokenizerWords":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(alphabet=data["alphabet"], eow=data["eow"])
        obj.vocab = {int(k): v for k, v in data["vocab"].items()}
        obj.inverse_vocab = {v: int(k) for k, v in data["vocab"].items()}
        obj.bpe_merges = {
            tuple(m["pair"]): m["new_id"] for m in data["bpe_merges"]
        }
        obj._build_merge_priority()
        return obj

    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most", exclude_id=None):
        pairs = Counter(
            (a, b) for a, b in zip(token_ids, token_ids[1:])
            if exclude_id is None or (a != exclude_id and b != exclude_id)
        )

        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        replaced = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair_id:
                replaced.append(new_id)
                i += 2
            else:
                replaced.append(token_ids[i])
                i += 1
        return replaced


def main():

    train_word = "babaBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBababaBABabaBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbababbabaBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbabAABABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbababAbabaBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBababaBABabaBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabABABababaBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbabAABABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbababA#babababab#aBababaBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbababABABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbababbabaBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbabaababaBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbababb#"
    parse_word = "babababab"
#    parse_word = "aBababaBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbababABABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbababbabaBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbabaababaBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbaBBABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbaBBABaBAbabABaBBAbabABABaBAbaBABaBBAbabbAABBABaBAbabABaBBAbabaBABaBAbaBABaBBAbabbABaBAbabABaBBAbabABABaBAbaBABaBBAbabbababb"

    tokenizer = BPETokenizerWords(['a','b','A','B'])
    word_list = [train_word]
    tokenizer.train(word_list, 1000)
    tokenizer.save("ml/bpe_model.json")

    print(tokenizer.vocab)
    tokens = tokenizer.encode(parse_word)
    print("Tokens:", tokens)
    words = tokenizer.decode(tokens)

    if words == parse_word:
        print("Success")
    else:
        print("Failed")

    print(f"Orig word len: {len(parse_word)}, tokens: {len(tokens)}")

    tokenizer2 = BPETokenizerWords.from_file("ml/bpe_model.json")
    tokens2 = tokenizer2.encode(parse_word)
    words2 = tokenizer2.decode(tokens2)
    if words2 == parse_word:
        print("Success (from_file)")
    else:
        print("Failed (from_file)")

if __name__ == "__main__":
    main()

