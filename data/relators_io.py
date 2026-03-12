"""
I/O utilities for relators data.
Load and convert between integer and string representations.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

import json
from typing import Dict, Optional

from algorithms.word_conversion import word_to_string, string_to_word


def load_relators(filepath: str) -> Dict:
    """Load relators from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dict with relator data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If invalid JSON
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def save_relators(filepath: str, data: dict):
    """Save relators into JSON file.

    Args:
        filepath: Path to JSON file
        data: Dict with relator data
    """
    with open(filepath, 'w') as f:
        json.dump(data, f)

def convert_to_strings(data: Dict) -> Dict:
    """Convert all word arrays to strings.

    Args:
        data: Dict with format {"R1": {"version": 1, "word": [...]}, ...}

    Returns:
        Dict with strings: {"R1": {"version": 1, "word": "ABa..."}, ...}
    """
    result = {}
    for key, value in data.items():
        result[key] = {
            "version": value["version"],
            "word": word_to_string(value["word"])
        }
        # Preserve any additional fields
        for k, v in value.items():
            if k not in ["version", "word"]:
                result[key][k] = v
    return result


def convert_to_integers(
    string_data: Dict,
    original_data: Dict,
    details: Optional[Dict[str, str]] = None
) -> Dict:
    """Convert strings back to integer arrays.

    Args:
        string_data: Dict with string words
        original_data: Original dict for version comparison
        details: Optional details messages {"R1": "simplified...", ...}

    Returns:
        Dict with integer arrays, incremented versions if shortened
    """
    result = {}
    for key, value in string_data.items():
        new_word = string_to_word(value["word"])

        # Get original length for comparison
        original_length = len(original_data[key]["word"])
        new_length = len(new_word)

        # Increment version if word shortened
        version = value["version"]
        if new_length < original_length:
            version += 1

        result[key] = {
            "version": version,
            "word": new_word
        }

        # Add details if provided
        if details and key in details:
            result[key]["details"] = details[key]

        # Preserve any additional fields
        for k, v in value.items():
            if k not in ["version", "word"]:
                result[key][k] = v

    return result
