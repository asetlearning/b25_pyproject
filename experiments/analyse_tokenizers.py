tokenizer_name = "models/tokenizer/bpe_tokenizer_reduced_relators_20260305.json"

import sys

# Add the directory to sys.path
sys.path.append('../b25_pyproject/')

from algorithms.word_compression import BPETokenizerWordCompression
from algorithms.maximal_power import find_maximal_powers
from algorithms.word_squares import find_squares

tokenizer = BPETokenizerWordCompression.from_file(tokenizer_name)

resuts = []
for t, v in tokenizer.vocab.items():
    squares = find_squares(v, f"T{t}")
    bases = {sq[:len(sq) // 2] for sq in squares}
    has_power = False
    for base in bases:
        if base:
#            print(" ***** ", base, v)
            powers = find_maximal_powers(base, v)
            for b, pow, n in powers:
                if pow > 2.5:
                    info = f" ***** {t}: pow: {pow}, tok: {len(v)} base:{len(b)}"
                    print(info)
                    resuts.append(info)
                    has_power = True
                    break
        else:
            print(t, "Empty base")
        if has_power:
            break

print("\n".join(resuts))