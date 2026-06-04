"""Character alphabet handling.

The model operates over a fixed, ordered alphabet of single characters.  The
alphabet bounds the size of the dense n-gram tables (``A^n`` entries), so it is
typically *reduced* to the most frequent characters in the training corpus via
:func:`select_alphabet`.

:class:`Alphabet` is an immutable value object.  Every character-to-index
lookup is bounds-checked so that neither a hostile corpus nor a hostile model
file can drive an out-of-range table access downstream.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from omen.errors import ConfigError, TrainingError

# Hard cap on alphabet size.  Dense tables are ``A^n``; this keeps a crafted or
# accidental value from requesting an enormous allocation.  256 comfortably
# covers printable ASCII plus common Latin-1 symbols.
MAX_ALPHABET_SIZE = 256


@dataclass(frozen=True, slots=True)
class Alphabet:
    """An immutable, ordered set of single-character symbols.

    ``chars[i]`` is the character with index ``i``; ``index[c]`` is the inverse.
    Construct via :meth:`from_chars` so the index map is built and validated.
    """

    chars: tuple[str, ...]
    index: Mapping[str, int]

    @classmethod
    def from_chars(cls, chars: Iterable[str]) -> Alphabet:
        """Build an alphabet from an ordered iterable of distinct characters."""
        ordered = tuple(chars)
        if not ordered:
            raise ConfigError("alphabet must not be empty")
        if len(ordered) > MAX_ALPHABET_SIZE:
            raise ConfigError(f"alphabet size {len(ordered)} exceeds maximum {MAX_ALPHABET_SIZE}")
        for ch in ordered:
            if len(ch) != 1:
                raise ConfigError(f"alphabet entries must be single characters, got {ch!r}")
        mapping = {ch: i for i, ch in enumerate(ordered)}
        if len(mapping) != len(ordered):
            raise ConfigError("alphabet contains duplicate characters")
        return cls(ordered, MappingProxyType(mapping))

    @property
    def size(self) -> int:
        """Number of symbols in the alphabet (``A``)."""
        return len(self.chars)

    def contains(self, ch: str) -> bool:
        """Return whether ``ch`` is part of this alphabet."""
        return ch in self.index

    def encode(self, text: str) -> list[int] | None:
        """Map ``text`` to indices, or return ``None`` if any char is foreign.

        Returning ``None`` lets callers skip whole passwords containing
        out-of-alphabet characters rather than fabricating misleading n-grams
        from the surviving fragments.
        """
        out: list[int] = []
        idx = self.index
        for ch in text:
            code = idx.get(ch)
            if code is None:
                return None
            out.append(code)
        return out

    def decode(self, codes: Iterable[int]) -> str:
        """Map indices back to a string, bounds-checking each index."""
        chars = self.chars
        size = len(chars)
        out: list[str] = []
        for code in codes:
            if not 0 <= code < size:
                raise ModelIndexError(code, size)
            out.append(chars[code])
        return "".join(out)

    def as_string(self) -> str:
        """Serialise the alphabet to a plain string for persistence."""
        return "".join(self.chars)


class ModelIndexError(ConfigError):
    """Raised when an alphabet index is outside the valid range."""

    def __init__(self, code: int, size: int) -> None:
        super().__init__(f"alphabet index {code} out of range [0, {size})")


@dataclass(frozen=True, slots=True)
class AlphabetSelection:
    """Result of :func:`select_alphabet`: the alphabet plus coverage stats."""

    alphabet: Alphabet
    coverage: float
    """Fraction of corpus characters retained by the chosen alphabet (0..1)."""
    total_chars: int
    distinct_chars: int


def select_alphabet(passwords: Iterable[str], size: int) -> AlphabetSelection:
    """Choose the ``size`` most frequent characters across ``passwords``.

    Ties are broken by Unicode code point so the result is deterministic.  The
    returned coverage is the share of all observed characters that the selected
    alphabet retains — a quick signal for whether ``size`` is large enough.
    """
    if size < 1:
        raise TrainingError("alphabet size must be at least 1")
    if size > MAX_ALPHABET_SIZE:
        raise TrainingError(f"alphabet size {size} exceeds maximum {MAX_ALPHABET_SIZE}")

    counts: Counter[str] = Counter()
    for pw in passwords:
        counts.update(pw)

    total = sum(counts.values())
    if total == 0:
        raise TrainingError("corpus contains no characters to build an alphabet from")

    # Sort by descending frequency, then ascending code point for determinism.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = ranked[:size]
    alphabet = Alphabet.from_chars(ch for ch, _ in chosen)
    retained = sum(count for _, count in chosen)
    return AlphabetSelection(
        alphabet=alphabet,
        coverage=retained / total,
        total_chars=total,
        distinct_chars=len(counts),
    )
