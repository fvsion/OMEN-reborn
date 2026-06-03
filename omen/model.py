"""The trained n-gram model and its on-disk format.

:class:`NgramModel` is the central data + query object.  It owns the alphabet,
the shared :class:`~omen.levels.LevelScale`, and four level tables:

* ``ip`` — initial level per ``(n-1)``-gram context (one byte each).
* ``cp`` — conditional level per ``(context, next-char)`` pair (dense, ``A^n``).
* ``ep`` — ending level per context (probability the word ends there).
* ``ln`` — length level per password length.

The total level of a password is the sum of its initial, conditional, ending,
and length levels — see :meth:`NgramModel.total_level`.  The enumerator emits
each candidate at exactly this total, so scoring and emission always agree.

Training lives in :mod:`omen.train`; this module is data + persistence + query.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from omen.alphabet import Alphabet
from omen.errors import ConfigError, ModelError
from omen.levels import LevelScale

CONFIG_NAME = "config.json"
IP_NAME = "ip.dat"
CP_NAME = "cp.dat"
EP_NAME = "ep.dat"

FORMAT_VERSION = 1

# Safety bounds applied when loading an untrusted model directory.
MAX_NGRAM = 5
MAX_PASSWORD_LENGTH = 256
# Hard cap on dense-table entries (``A^n``) to prevent a crafted config from
# requesting a huge allocation.  2**28 bytes = 256 MiB.
MAX_TABLE_ENTRIES = 1 << 28


def checked_pow(base: int, exp: int, what: str) -> int:
    """Compute ``base ** exp``, raising :class:`ConfigError` if it is too large."""
    result: int = base**exp
    if result > MAX_TABLE_ENTRIES:
        raise ConfigError(
            f"{what} would require {result} entries, exceeding the limit {MAX_TABLE_ENTRIES}"
        )
    return result


class NgramModel:
    """A trained order-``n`` Markov model with discretised level tables."""

    __slots__ = (
        "_a",
        "_cp_stride",
        "_drop_mod",
        "alphabet",
        "context_len",
        "coverage",
        "cp_levels",
        "ep_enabled",
        "ep_levels",
        "ip_levels",
        "ln_levels",
        "max_length",
        "ngram",
        "scale",
        "smoothing",
    )

    def __init__(
        self,
        *,
        alphabet: Alphabet,
        scale: LevelScale,
        ngram: int,
        smoothing: float,
        max_length: int,
        ip_levels: bytes,
        cp_levels: bytes,
        ep_levels: bytes,
        ep_enabled: bool,
        ln_levels: Sequence[int],
        coverage: float,
    ) -> None:
        if ngram < 2:
            raise ConfigError(f"ngram must be >= 2, got {ngram}")
        if ngram > MAX_NGRAM:
            raise ConfigError(f"ngram {ngram} exceeds maximum {MAX_NGRAM}")

        self.alphabet = alphabet
        self.scale = scale
        self.ngram = ngram
        self.smoothing = smoothing
        self.max_length = max_length
        self.ep_enabled = ep_enabled
        self.coverage = coverage

        self._a = alphabet.size
        self.context_len = ngram - 1
        num_contexts = checked_pow(self._a, self.context_len, "ip/ep table")
        cp_entries = checked_pow(self._a, ngram, "cp table")
        self._cp_stride = self._a
        # Used to shift a context window: drop the oldest symbol, append a new one.
        self._drop_mod = checked_pow(self._a, self.context_len - 1, "context shift")

        self._require_size(ip_levels, num_contexts, "ip")
        self._require_size(cp_levels, cp_entries, "cp")
        self._require_size(ep_levels, num_contexts, "ep")
        self.ip_levels = ip_levels
        self.cp_levels = cp_levels
        self.ep_levels = ep_levels

        if max_length < self.context_len or max_length > MAX_PASSWORD_LENGTH:
            raise ConfigError(
                f"max_length {max_length} out of range [{self.context_len}, {MAX_PASSWORD_LENGTH}]"
            )
        if len(ln_levels) != max_length + 1:
            raise ConfigError(f"ln table has {len(ln_levels)} entries, expected {max_length + 1}")
        max_lvl = scale.max_level
        for length, lvl in enumerate(ln_levels):
            if not 0 <= lvl <= max_lvl:
                raise ConfigError(f"ln level for length {length} out of range: {lvl}")
        self.ln_levels = tuple(ln_levels)

    @staticmethod
    def _require_size(table: bytes, expected: int, name: str) -> None:
        if len(table) != expected:
            raise ConfigError(f"{name} table has {len(table)} bytes, expected {expected}")

    # -- properties --------------------------------------------------------

    @property
    def alphabet_size(self) -> int:
        """Number of symbols in the alphabet (``A``)."""
        return self._a

    @property
    def min_length(self) -> int:
        """Shortest representable password (one full initial context)."""
        return self.context_len

    @property
    def num_contexts(self) -> int:
        """Number of distinct ``(n-1)``-gram contexts (``A^(n-1)``)."""
        return len(self.ip_levels)

    # -- query primitives --------------------------------------------------

    def context_index(self, codes: Sequence[int]) -> int:
        """Pack ``context_len`` alphabet indices into a single context index."""
        idx = 0
        for code in codes:
            idx = idx * self._a + code
        return idx

    def next_context(self, ctx: int, code: int) -> int:
        """Slide the context window: drop the oldest symbol, append ``code``."""
        return (ctx % self._drop_mod) * self._a + code

    def ip_level(self, ctx: int) -> int:
        """Initial level of context ``ctx``."""
        return self.ip_levels[ctx]

    def ep_level(self, ctx: int) -> int:
        """Ending level of context ``ctx`` (0 when EP is disabled)."""
        return self.ep_levels[ctx] if self.ep_enabled else 0

    def cp_level(self, ctx: int, code: int) -> int:
        """Conditional level of emitting ``code`` after context ``ctx``."""
        return self.cp_levels[ctx * self._cp_stride + code]

    def total_level(self, codes: Sequence[int]) -> int | None:
        """Total level of a full password given as alphabet indices.

        Returns ``None`` when the length is outside ``[min_length, max_length]``.
        This is the single source of truth for a candidate's level; the
        enumerator emits each candidate at exactly this value.
        """
        length = len(codes)
        if length < self.context_len or length > self.max_length:
            return None
        ctx = self.context_index(codes[: self.context_len])
        total = self.ip_level(ctx)
        for i in range(self.context_len, length):
            code = codes[i]
            total += self.cp_level(ctx, code)
            ctx = self.next_context(ctx, code)
        total += self.ep_level(ctx)
        total += self.ln_levels[length]
        return total

    # -- persistence -------------------------------------------------------

    def save(self, directory: Path | str) -> None:
        """Write the model to ``directory`` (created if absent)."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "version": FORMAT_VERSION,
            "ngram": self.ngram,
            "levels": self.scale.levels,
            "lam": self.scale.lam,
            "smoothing": self.smoothing,
            "max_length": self.max_length,
            "alphabet": self.alphabet.as_string(),
            "ep_enabled": self.ep_enabled,
            "ln_levels": list(self.ln_levels),
            "coverage": self.coverage,
        }
        (path / CONFIG_NAME).write_text(json.dumps(config, indent=2), encoding="utf-8")
        (path / IP_NAME).write_bytes(self.ip_levels)
        (path / CP_NAME).write_bytes(self.cp_levels)
        (path / EP_NAME).write_bytes(self.ep_levels)

    @classmethod
    def load(cls, directory: Path | str) -> NgramModel:
        """Load and validate a model from ``directory``.

        The config is parsed and every field range-checked *before* any table
        file is read, so a malformed or hostile model cannot trigger a large
        allocation or an out-of-range table access.
        """
        path = Path(directory)
        config_path = path / CONFIG_NAME
        if not config_path.is_file():
            raise ModelError(f"no {CONFIG_NAME} found in {path}")
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ConfigError(f"could not read {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError("config root must be a JSON object")

        version = _require_int(raw, "version")
        if version != FORMAT_VERSION:
            raise ConfigError(f"unsupported model version {version} (expected {FORMAT_VERSION})")

        ngram = _require_int(raw, "ngram")
        levels = _require_int(raw, "levels")
        lam = _require_float(raw, "lam")
        smoothing = _require_float(raw, "smoothing")
        max_length = _require_int(raw, "max_length")
        alphabet_str = _require_str(raw, "alphabet")
        ep_enabled = _require_bool(raw, "ep_enabled")
        coverage = _require_float(raw, "coverage")
        ln_raw = raw.get("ln_levels")
        if not isinstance(ln_raw, list) or not all(isinstance(x, int) for x in ln_raw):
            raise ConfigError("ln_levels must be a list of integers")

        alphabet = Alphabet.from_chars(alphabet_str)
        scale = LevelScale(levels=levels, lam=lam)  # validates levels/lam ranges

        ip_levels = _read_table(path / IP_NAME)
        cp_levels = _read_table(path / CP_NAME)
        ep_levels = _read_table(path / EP_NAME)

        # The constructor performs the remaining cross-field validation
        # (table sizes vs. A^n, ln length, level ranges) and raises ConfigError.
        return cls(
            alphabet=alphabet,
            scale=scale,
            ngram=ngram,
            smoothing=smoothing,
            max_length=max_length,
            ip_levels=ip_levels,
            cp_levels=cp_levels,
            ep_levels=ep_levels,
            ep_enabled=ep_enabled,
            ln_levels=ln_raw,
            coverage=coverage,
        )


def _read_table(path: Path) -> bytes:
    if not path.is_file():
        raise ModelError(f"missing table file {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ModelError(f"could not read {path}: {exc}") from exc


def _require_int(obj: dict[str, object], key: str) -> int:
    value = obj.get(key)
    # bool is a subclass of int; reject it where a real integer is expected.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"config field {key!r} must be an integer")
    return value


def _require_float(obj: dict[str, object], key: str) -> float:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"config field {key!r} must be a number")
    return float(value)


def _require_str(obj: dict[str, object], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str):
        raise ConfigError(f"config field {key!r} must be a string")
    return value


def _require_bool(obj: dict[str, object], key: str) -> bool:
    value = obj.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"config field {key!r} must be a boolean")
    return value
