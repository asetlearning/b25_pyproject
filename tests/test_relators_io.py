"""
Tests for relators I/O utilities.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

import pytest
import json
import tempfile
import os

from data.relators_io import (
    load_relators,
    convert_to_strings,
    convert_to_integers
)


@pytest.fixture
def sample_data():
    """Sample relator data for testing."""
    return {
        "R1": {"version": 1, "word": [-1, -2, 1, -2, -1, 2, 1]},
        "R2": {"version": 1, "word": [1, 2, -1, -2]},
        "R3": {"version": 2, "word": [-1, 1]}
    }


@pytest.fixture
def temp_json_file(sample_data):
    """Create temporary JSON file with sample data."""
    fd, path = tempfile.mkstemp(suffix='.json')
    with os.fdopen(fd, 'w') as f:
        json.dump(sample_data, f)
    yield path
    os.unlink(path)


def test_load_relators(temp_json_file, sample_data):
    """Test loading relators from JSON file."""
    data = load_relators(temp_json_file)
    assert data == sample_data
    assert "R1" in data
    assert "version" in data["R1"]
    assert "word" in data["R1"]


def test_load_relators_file_not_found():
    """Test loading non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_relators("/nonexistent/path.json")


def test_convert_to_strings(sample_data):
    """Test converting words to strings."""
    result = convert_to_strings(sample_data)

    assert result["R1"]["word"] == "ABaBAba"
    assert result["R1"]["version"] == 1
    assert result["R2"]["word"] == "abAB"
    assert result["R2"]["version"] == 1
    assert result["R3"]["word"] == "Aa"
    assert result["R3"]["version"] == 2


def test_convert_to_integers(sample_data):
    """Test converting strings back to integers."""
    string_data = convert_to_strings(sample_data)
    result = convert_to_integers(string_data, sample_data)

    assert result["R1"]["word"] == sample_data["R1"]["word"]
    assert result["R1"]["version"] == 1
    assert result["R2"]["word"] == sample_data["R2"]["word"]
    assert result["R3"]["word"] == sample_data["R3"]["word"]


def test_version_increment(sample_data):
    """Test version increments when word shortened."""
    string_data = convert_to_strings(sample_data)

    # Shorten R1 word
    string_data["R1"]["word"] = "AB"  # Shorter than original

    result = convert_to_integers(string_data, sample_data)

    # Version should increment because word shortened
    assert result["R1"]["version"] == 2
    assert result["R1"]["word"] == [-1, -2]


def test_version_no_change(sample_data):
    """Test version unchanged when length same."""
    string_data = convert_to_strings(sample_data)

    # Change R1 word but keep same length
    original_length = len(sample_data["R1"]["word"])
    string_data["R1"]["word"] = "a" * original_length

    result = convert_to_integers(string_data, sample_data)

    # Version should stay same
    assert result["R1"]["version"] == 1


def test_version_no_change_longer(sample_data):
    """Test version unchanged when word longer."""
    string_data = convert_to_strings(sample_data)

    # Lengthen R1 word
    string_data["R1"]["word"] = "ABaBAbaaaa"  # Longer

    result = convert_to_integers(string_data, sample_data)

    # Version should stay same (only shortening increments)
    assert result["R1"]["version"] == 1


def test_details_field(sample_data):
    """Test adding details field."""
    string_data = convert_to_strings(sample_data)
    string_data["R1"]["word"] = "AB"  # Shorten it

    details = {
        "R1": "simplified by removing redundancy",
        "R2": "no changes"
    }

    result = convert_to_integers(string_data, sample_data, details)

    assert result["R1"]["details"] == "simplified by removing redundancy"
    assert result["R2"]["details"] == "no changes"
    assert "details" not in result["R3"]  # No details for R3


def test_full_roundtrip(temp_json_file):
    """Test full roundtrip: load -> strings -> integers."""
    # Load original data
    original = load_relators(temp_json_file)

    # Convert to strings
    strings = convert_to_strings(original)

    # Convert back to integers
    result = convert_to_integers(strings, original)

    # Should match original (no shortening)
    for key in original:
        assert result[key]["word"] == original[key]["word"]
        assert result[key]["version"] == original[key]["version"]


def test_preserve_additional_fields(sample_data):
    """Test that additional fields are preserved."""
    # Add extra field
    sample_data["R1"]["metadata"] = {"author": "test"}

    string_data = convert_to_strings(sample_data)
    assert string_data["R1"]["metadata"] == {"author": "test"}

    result = convert_to_integers(string_data, sample_data)
    assert result["R1"]["metadata"] == {"author": "test"}


def test_actual_data_file():
    """Test with actual data file if it exists."""
    filepath = "data/b25_nontrivial_relators_update_azsw1h6x_20260129.json"

    if not os.path.exists(filepath):
        pytest.skip("Actual data file not found")

    data = load_relators(filepath)
    assert isinstance(data, dict)
    assert len(data) > 0

    # Convert to strings and back
    strings = convert_to_strings(data)
    assert isinstance(strings, dict)

    # Verify R1 if it exists
    if "R1" in strings:
        assert isinstance(strings["R1"]["word"], str)
