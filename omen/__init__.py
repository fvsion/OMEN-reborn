"""OMEN-fvsion — a clean-room Ordered Markov ENumerator.

Builds an order-``n`` Markov model from a password corpus and emits candidate
passwords in (approximately) descending probability order, suitable for piping
into a password-recovery back-end such as hashcat.

This is an independent implementation written from the published algorithm
(Dürmuth et al., ESSoS 2015); no original OMEN source is used.
"""

from __future__ import annotations

from omen.alphabet import Alphabet, AlphabetSelection, select_alphabet
from omen.enumerate import Enumerator, PyEnumerator
from omen.errors import ConfigError, ModelError, OmenError, TrainingError
from omen.levels import LevelScale
from omen.model import NgramModel
from omen.score import PasswordScore, PasswordScorer
from omen.train import ModelTrainer, TrainingOptions

__version__ = "0.1.0"

__all__ = [
    "Alphabet",
    "AlphabetSelection",
    "ConfigError",
    "Enumerator",
    "LevelScale",
    "ModelError",
    "ModelTrainer",
    "NgramModel",
    "OmenError",
    "PasswordScore",
    "PasswordScorer",
    "PyEnumerator",
    "TrainingError",
    "TrainingOptions",
    "__version__",
    "select_alphabet",
]
