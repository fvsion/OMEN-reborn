"""Tests for the ordered enumerator."""

from __future__ import annotations

from omen.enumerate import Enumerator, PyEnumerator
from omen.model import NgramModel
from omen.train import ModelTrainer, TrainingOptions


class RecordingSink:
    """Collects emitted candidates as decoded strings, in order."""

    def __init__(self) -> None:
        self.items: list[str] = []

    def write_line_bytes(self, raw: bytes) -> None:
        self.items.append(raw.decode("utf-8"))


def _tiny_model() -> NgramModel:
    options = TrainingOptions(ngram=2, alphabet="ab", levels=8, max_length=3)
    corpus = ["a", "b", "ab", "ba", "aab", "bba", "abb"]
    return ModelTrainer(options).train(lambda: iter(corpus))


def test_pyenumerator_satisfies_protocol(trained_model: NgramModel) -> None:
    assert isinstance(PyEnumerator(trained_model), Enumerator)


def test_output_is_non_decreasing_in_level(trained_model: NgramModel) -> None:
    sink = RecordingSink()
    PyEnumerator(trained_model).stream(sink, max_guesses=5000)
    totals = []
    for candidate in sink.items:
        codes = trained_model.alphabet.encode(candidate)
        assert codes is not None  # every emitted char must be in the alphabet
        total = trained_model.total_level(codes)
        assert total is not None
        totals.append(total)
    assert totals == sorted(totals), "levels must be emitted in non-decreasing order"


def test_no_duplicate_candidates(trained_model: NgramModel) -> None:
    sink = RecordingSink()
    PyEnumerator(trained_model).stream(sink, max_guesses=5000)
    assert len(sink.items) == len(set(sink.items))


def test_full_enumeration_covers_keyspace() -> None:
    model = _tiny_model()
    sink = RecordingSink()
    PyEnumerator(model).stream(sink)
    # All strings over {a,b} of length 1..3: 2 + 4 + 8 = 14, each exactly once.
    expected = set()
    for length in (1, 2, 3):
        for n in range(2**length):
            expected.add("".join("ab"[(n >> i) & 1] for i in range(length)))
    assert set(sink.items) == expected
    assert len(sink.items) == len(expected) == 14


def test_max_guesses_is_respected(trained_model: NgramModel) -> None:
    sink = RecordingSink()
    count = PyEnumerator(trained_model).stream(sink, max_guesses=10)
    assert count == 10
    assert len(sink.items) == 10


def test_length_filter(trained_model: NgramModel) -> None:
    sink = RecordingSink()
    PyEnumerator(trained_model).stream(sink, max_guesses=2000, min_length=8, max_length=8)
    assert sink.items, "expected some length-8 candidates"
    assert all(len(item) == 8 for item in sink.items)


def test_common_passwords_surface_early(trained_model: NgramModel) -> None:
    sink = RecordingSink()
    PyEnumerator(trained_model).stream(sink, max_guesses=200_000)
    head = set(sink.items[:20_000])
    # Frequent training passwords should appear well within the early stream.
    assert "password" in head
    assert "123456" in head
