"""Probability-to-level discretisation.

OMEN orders candidates by an integer *level* rather than a floating-point
probability.  A probability ``p`` maps to a level via

    level(p) = clamp(round(-ln(p) / lam), 0, NL - 1)

so that the most probable events land at level 0 and the least probable at
``NL - 1``.  ``lam`` is the resolution: nats of surprisal per level.

A single :class:`LevelScale` is shared by every table in a model (initial,
conditional, ending, length).  Because levels are summed across the components
of a candidate, they must share one scale for the total to be meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omen.errors import ConfigError

# Levels are stored as unsigned bytes on disk, so the count must fit in a byte.
MAX_LEVELS = 256


@dataclass(frozen=True, slots=True)
class LevelScale:
    """Immutable mapping from probability to discrete level.

    Attributes:
        levels: Number of distinct levels ``NL`` (``>= 2``).
        lam: Surprisal resolution in nats per level (``> 0``).
    """

    levels: int
    lam: float

    def __post_init__(self) -> None:
        if self.levels < 2:
            raise ConfigError(f"levels must be >= 2, got {self.levels}")
        if self.levels > MAX_LEVELS:
            raise ConfigError(f"levels {self.levels} exceeds maximum {MAX_LEVELS}")
        if not (self.lam > 0.0 and math.isfinite(self.lam)):
            raise ConfigError(f"lam must be a positive finite float, got {self.lam}")

    @property
    def max_level(self) -> int:
        """The worst (least probable) level, ``NL - 1``."""
        return self.levels - 1

    def to_level(self, prob: float) -> int:
        """Discretise a probability into a level in ``[0, NL - 1]``.

        Non-positive probabilities (which should not occur after smoothing) map
        to the worst level rather than raising, so the function is total.
        """
        if prob <= 0.0:
            return self.max_level
        surprisal = -math.log(prob)
        level = round(surprisal / self.lam)
        if level < 0:
            return 0
        if level > self.max_level:
            return self.max_level
        return level

    @classmethod
    def from_floor(cls, levels: int, floor_prob: float) -> LevelScale:
        """Build a scale where ``floor_prob`` maps onto the worst level.

        ``floor_prob`` is the smallest probability the model will ever
        discretise (the most heavily smoothed entry).  Choosing ``lam`` from it
        spreads the observed probability range across all available levels.
        """
        if not (0.0 < floor_prob < 1.0):
            raise ConfigError(f"floor probability must be in (0, 1), got {floor_prob}")
        if levels < 2:
            raise ConfigError(f"levels must be >= 2, got {levels}")
        lam = -math.log(floor_prob) / (levels - 1)
        return cls(levels=levels, lam=lam)
