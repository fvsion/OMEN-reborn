"""Tests for the probability-to-level discretisation."""

from __future__ import annotations

import math

import pytest

from omen.errors import ConfigError
from omen.levels import LevelScale


def test_more_probable_maps_to_lower_or_equal_level() -> None:
    scale = LevelScale.from_floor(levels=11, floor_prob=1e-6)
    probs = [0.9, 0.5, 0.1, 0.01, 1e-3, 1e-5, 1e-6]
    levels = [scale.to_level(p) for p in probs]
    # Monotonic non-decreasing as probability decreases.
    assert levels == sorted(levels)


def test_bounds_are_respected() -> None:
    scale = LevelScale.from_floor(levels=8, floor_prob=1e-4)
    assert scale.to_level(1.0) == 0
    assert scale.to_level(1e-4) == scale.max_level
    # Below the floor still clamps to the worst level, never beyond.
    assert scale.to_level(1e-9) == scale.max_level
    assert scale.to_level(0.0) == scale.max_level


def test_floor_maps_exactly_to_max_level() -> None:
    floor = 1e-5
    scale = LevelScale.from_floor(levels=11, floor_prob=floor)
    assert scale.to_level(floor) == scale.max_level
    # lam is surprisal-per-level derived from the floor.
    assert math.isclose(scale.lam, -math.log(floor) / (scale.levels - 1))


@pytest.mark.parametrize("levels", [0, 1, 257])
def test_invalid_level_count_rejected(levels: int) -> None:
    with pytest.raises(ConfigError):
        LevelScale(levels=levels, lam=1.0)


@pytest.mark.parametrize("lam", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_lam_rejected(lam: float) -> None:
    with pytest.raises(ConfigError):
        LevelScale(levels=11, lam=lam)


@pytest.mark.parametrize("floor", [0.0, 1.0, 1.5, -0.1])
def test_invalid_floor_rejected(floor: float) -> None:
    with pytest.raises(ConfigError):
        LevelScale.from_floor(levels=11, floor_prob=floor)
