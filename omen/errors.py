"""Typed exception hierarchy.

Every error the package raises for an anticipated, user-facing condition is an
:class:`OmenError`.  The CLI catches this base class and prints a clean,
actionable message instead of a traceback.
"""

from __future__ import annotations


class OmenError(Exception):
    """Base class for all anticipated OMEN errors."""


class TrainingError(OmenError):
    """Raised when a model cannot be trained from the supplied corpus/options."""


class ModelError(OmenError):
    """Raised when a model is structurally invalid or cannot be loaded."""


class ConfigError(ModelError):
    """Raised when a model's ``config.json`` is missing fields or out of range.

    A dedicated subclass because configuration is the primary trust boundary
    for an on-disk model: it is validated before any table is allocated.
    """
