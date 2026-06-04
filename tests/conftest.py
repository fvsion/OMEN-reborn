"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from omen.model import NgramModel
from omen.train import ModelTrainer, TrainingOptions

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_passwords.txt"


def _read_sample() -> list[str]:
    return [
        line.rstrip("\r\n")
        for line in SAMPLE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="session")
def sample_passwords() -> list[str]:
    return _read_sample()


@pytest.fixture(scope="session")
def trained_model(sample_passwords: list[str]) -> NgramModel:
    options = TrainingOptions(ngram=3, levels=11, max_length=16, alphabet_size=72)
    return ModelTrainer(options).train(lambda: iter(sample_passwords))
