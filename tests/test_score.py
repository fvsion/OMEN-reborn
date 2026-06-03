"""Tests for password scoring."""

from __future__ import annotations

from omen.enumerate import PyEnumerator
from omen.model import NgramModel
from omen.score import PasswordScorer


class _RecordingSink:
    def __init__(self) -> None:
        self.items: list[str] = []

    def write_line_bytes(self, raw: bytes) -> None:
        self.items.append(raw.decode("utf-8"))


def test_score_matches_emission_level(trained_model: NgramModel) -> None:
    """A scored password's level equals the level the enumerator emits it at."""
    scorer = PasswordScorer(trained_model)
    sink = _RecordingSink()
    PyEnumerator(trained_model).stream(sink, max_guesses=3000)
    for candidate in sink.items[:200]:
        result = scorer.score(candidate)
        codes = trained_model.alphabet.encode(candidate)
        assert codes is not None
        assert result.in_model
        assert result.total_level == trained_model.total_level(codes)


def test_components_sum_to_total(trained_model: NgramModel) -> None:
    result = PasswordScorer(trained_model).score("password")
    assert result.in_model
    assert result.total_level == (
        (result.ip_level or 0)
        + (result.cp_sum or 0)
        + (result.ep_level or 0)
        + (result.ln_level or 0)
    )


def test_out_of_alphabet_is_unreachable(trained_model: NgramModel) -> None:
    # A character essentially never in the sample alphabet.
    result = PasswordScorer(trained_model).score("pass☃word")
    assert not result.in_model
    assert result.total_level is None
    assert result.reason is not None


def test_rank_estimate_is_monotonic(trained_model: NgramModel) -> None:
    scorer = PasswordScorer(trained_model)
    common = scorer.estimate_rank("password", cap=50_000)
    rare = scorer.estimate_rank("zxqvbn", cap=50_000)
    assert common is not None and rare is not None
    # A more probable password should not rank later than a less probable one.
    assert common <= rare
