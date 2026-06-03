"""Password scoring — the clean replacement for OMEN's ``evalPW``.

:class:`PasswordScorer` reports the total level of a given password using the
exact same computation the enumerator uses to emit it, so a password's score
always equals the level at which the enumerator would produce it.

It can also *estimate* a guess rank: the number of candidates the enumerator
would emit before this password (i.e. at a strictly lower level), computed by
running the enumerator into a counting sink with an explicit cap.
"""

from __future__ import annotations

from dataclasses import dataclass

from omen.enumerate import PyEnumerator
from omen.model import NgramModel


@dataclass(frozen=True, slots=True)
class PasswordScore:
    """The result of scoring one password."""

    password: str
    in_model: bool
    total_level: int | None
    length: int
    reason: str | None = None
    ip_level: int | None = None
    cp_sum: int | None = None
    ep_level: int | None = None
    ln_level: int | None = None


class _CountingSink:
    """A :class:`~omen.io_utils.LineSink` that counts lines instead of writing."""

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def write_line_bytes(self, raw: bytes) -> None:
        self.count += 1


class PasswordScorer:
    """Scores passwords against a trained :class:`NgramModel`."""

    def __init__(self, model: NgramModel) -> None:
        self._model = model

    def score(self, password: str) -> PasswordScore:
        """Return the level breakdown for ``password``.

        ``in_model`` is ``False`` (with a human-readable ``reason``) when the
        password contains an out-of-alphabet character or its length falls
        outside ``[min_length, max_length]`` — such passwords are unreachable by
        the model and have no finite level.
        """
        model = self._model
        codes = model.alphabet.encode(password)
        if codes is None:
            return PasswordScore(
                password=password,
                in_model=False,
                total_level=None,
                length=len(password),
                reason="contains an out-of-alphabet character",
            )
        length = len(codes)
        if length < model.context_len or length > model.max_length:
            return PasswordScore(
                password=password,
                in_model=False,
                total_level=None,
                length=length,
                reason=f"length {length} outside [{model.min_length}, {model.max_length}]",
            )

        ctx = model.context_index(codes[: model.context_len])
        ip_level = model.ip_level(ctx)
        cp_sum = 0
        for i in range(model.context_len, length):
            code = codes[i]
            cp_sum += model.cp_level(ctx, code)
            ctx = model.next_context(ctx, code)
        ep_level = model.ep_level(ctx)
        ln_level = model.ln_levels[length]
        total = ip_level + cp_sum + ep_level + ln_level
        return PasswordScore(
            password=password,
            in_model=True,
            total_level=total,
            length=length,
            ip_level=ip_level,
            cp_sum=cp_sum,
            ep_level=ep_level,
            ln_level=ln_level,
        )

    def estimate_rank(self, password: str, cap: int = 10_000_000) -> int | None:
        """Estimate how many candidates precede ``password`` (capped at ``cap``).

        Returns the number of candidates with a strictly lower total level, or
        ``None`` if the password is not representable by the model.  The value
        is a lower bound on the true guess number (it excludes the password's
        position *within* its own level) and is capped to bound the work.
        """
        result = self.score(password)
        if not result.in_model or result.total_level is None:
            return None
        if result.total_level == 0:
            return 0
        sink = _CountingSink()
        PyEnumerator(self._model).stream(sink, max_guesses=cap, max_level=result.total_level - 1)
        return sink.count
