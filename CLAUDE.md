# Project: Burnside Problem B(2,5)

## Word Conversion System

Implemented Feb 10, 2026.

### Files Created

**Core utilities:**
- `algorithms/word_conversion.py` - Bidirectional conversion between integer arrays and strings
  - Mapping: 1→a, 2→b, -1→A, -2→B
  - Functions: `int_to_char()`, `word_to_string()`, `char_to_int()`, `string_to_word()`
- `data/relators_io.py` - JSON I/O for relator data
  - `load_relators()` - Load JSON file
  - `convert_to_strings()` - Convert word arrays to strings
  - `convert_to_integers()` - Convert back, auto-increment version if shortened, optional details field

**Tests:**
- `tests/test_word_conversion.py` - 10 tests incl. roundtrip, error handling
- `tests/test_relators_io.py` - 12 tests incl. version logic, actual data file

### Data Format

JSON structure: `{"R1": {"version": 1, "word": [...]}, ...}`
- 150 relators in `data/b25_nontrivial_relators_update_azsw1h6x_20260129.json`
- Version increments when word shortened
- Optional `details` field for change descriptions

### Usage

```python
from algorithms.word_conversion import word_to_string, string_to_word
from data.relators_io import load_relators, convert_to_strings, convert_to_integers

# Load and convert
data = load_relators('data/b25_nontrivial_relators_update_azsw1h6x_20260129.json')
strings = convert_to_strings(data)  # {"R1": {"version": 1, "word": "ABaBAba"}}

# Manipulate strings, convert back
strings["R1"]["word"] = "AB"  # Shortened
result = convert_to_integers(strings, data, details={"R1": "simplified"})
# result["R1"]["version"] == 2 (auto-incremented)
```

### Dependencies

Added to `requirements.txt`: `pytest>=7.0`
