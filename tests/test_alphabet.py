"""Tests for alphabet construction and selection."""

from __future__ import annotations

import pytest

from omen.alphabet import Alphabet, select_alphabet
from omen.errors import ConfigError, TrainingError


def test_from_chars_builds_inverse_index() -> None:
    alpha = Alphabet.from_chars("abc")
    assert alpha.size == 3
    assert alpha.index["a"] == 0
    assert alpha.index["c"] == 2
    assert alpha.encode("cab") == [2, 0, 1]
    assert alpha.decode([2, 0, 1]) == "cab"


def test_encode_returns_none_for_foreign_char() -> None:
    alpha = Alphabet.from_chars("abc")
    assert alpha.encode("axc") is None


def test_duplicate_and_multichar_rejected() -> None:
    with pytest.raises(ConfigError):
        Alphabet.from_chars("aab")
    with pytest.raises(ConfigError):
        Alphabet.from_chars(["ab", "c"])
    with pytest.raises(ConfigError):
        Alphabet.from_chars("")


def test_decode_bounds_checked() -> None:
    alpha = Alphabet.from_chars("abc")
    with pytest.raises(ConfigError):
        alpha.decode([5])


def test_select_alphabet_picks_most_frequent() -> None:
    passwords = ["aaaa", "aaab", "bbcc", "abcd"]
    selection = select_alphabet(passwords, size=2)
    assert set(selection.alphabet.chars) == {"a", "b"}
    assert 0.0 < selection.coverage <= 1.0
    assert selection.distinct_chars == 4


def test_select_alphabet_is_deterministic_on_ties() -> None:
    passwords = ["abcd"]  # all chars tie at frequency 1
    first = select_alphabet(passwords, size=2).alphabet.as_string()
    second = select_alphabet(passwords, size=2).alphabet.as_string()
    assert first == second == "ab"  # ties broken by code point


def test_select_alphabet_rejects_empty_corpus() -> None:
    with pytest.raises(TrainingError):
        select_alphabet([], size=4)
