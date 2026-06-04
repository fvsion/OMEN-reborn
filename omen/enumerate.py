"""Ordered candidate enumeration — the hot path.

:class:`PyEnumerator` walks the total level ``T = 0, 1, 2, …`` and, for each
``T``, emits every candidate whose total level equals ``T`` before advancing.
Total level is ``IP + Σ CP + EP + LN`` (see :meth:`NgramModel.total_level`), so
the output stream is globally non-decreasing in level — i.e. ordered from most
to least probable.

The :class:`Enumerator` Protocol is the seam for a future native (C/Cython)
implementation: it would consume the same flat level tables and satisfy the
same ``stream`` contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omen.io_utils import LineSink
from omen.model import NgramModel


@runtime_checkable
class Enumerator(Protocol):
    """Streams ranked password candidates into a :class:`LineSink`."""

    def stream(
        self,
        sink: LineSink,
        *,
        max_guesses: int | None = None,
        max_level: int | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> int:
        """Emit candidates and return how many were written."""
        ...


class _StopStreaming(Exception):
    """Internal control-flow signal: the guess budget has been reached."""


class PyEnumerator:
    """Pure-Python reference enumerator over an :class:`NgramModel`."""

    def __init__(self, model: NgramModel) -> None:
        self._model = model
        self._a = model.alphabet_size
        self._ctx_len = model.context_len
        self._drop_mod = self._a ** (self._ctx_len - 1)
        self._max_level = model.scale.max_level

        # Per-code UTF-8 bytes, so a candidate is assembled by byte concatenation.
        self._code_bytes: tuple[bytes, ...] = tuple(
            ch.encode("utf-8") for ch in model.alphabet.chars
        )

        # Group every context by its initial level: ip_buckets[level] -> [ctx, ...].
        self._ip_buckets: dict[int, list[int]] = {}
        for ctx, lvl in enumerate(model.ip_levels):
            self._ip_buckets.setdefault(lvl, []).append(ctx)

        # Global level bounds used to prune branches that cannot hit the budget.
        self._min_cp = min(model.cp_levels)
        self._max_cp = max(model.cp_levels)
        if model.ep_enabled:
            self._min_ep = min(model.ep_levels)
            self._max_ep = max(model.ep_levels)
        else:
            self._min_ep = 0
            self._max_ep = 0
        self._ip_max = max(model.ip_levels)

        # Lazily-built, cached per-context continuation buckets:
        #   ctx -> tuple of (cp_level, tuple_of_codes) sorted ascending by level.
        self._cp_cache: dict[int, tuple[tuple[int, tuple[int, ...]], ...]] = {}

        # Mutable per-stream state.
        self._emitted = 0
        self._limit: int | None = None

    # -- public API --------------------------------------------------------

    def stream(
        self,
        sink: LineSink,
        *,
        max_guesses: int | None = None,
        max_level: int | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> int:
        """Emit ranked candidates into ``sink``; return the number emitted.

        Candidates are produced in non-decreasing total-level order.  Within a
        single level the order is unspecified but deterministic.
        """
        model = self._model
        lo = model.min_length if min_length is None else max(min_length, model.min_length)
        hi = model.max_length if max_length is None else min(max_length, model.max_length)
        if lo > hi:
            return 0

        t_ceiling = self._total_level_ceiling(lo, hi)
        t_cap = t_ceiling if max_level is None else min(max_level, t_ceiling)

        self._emitted = 0
        self._limit = max_guesses
        if max_guesses is not None and max_guesses <= 0:
            return 0

        ln = model.ln_levels
        try:
            for total in range(t_cap + 1):
                for length in range(lo, hi + 1):
                    budget = total - ln[length]
                    if budget < 0:
                        continue
                    self._enumerate_length(length, budget, sink)
        except _StopStreaming:
            pass
        return self._emitted

    # -- enumeration core --------------------------------------------------

    def _enumerate_length(self, length: int, budget: int, sink: LineSink) -> None:
        """Emit every length-``length`` candidate whose n-gram total == ``budget``."""
        transitions = length - self._ctx_len
        tail_low = transitions * self._min_cp + self._min_ep
        tail_high = transitions * self._max_cp + self._max_ep

        # Feasible initial-level window: budget - a must land in [tail_low, tail_high].
        a_lo = max(0, budget - tail_high)
        a_hi = min(self._max_level, budget - tail_low)
        for a in range(a_lo, a_hi + 1):
            contexts = self._ip_buckets.get(a)
            if not contexts:
                continue
            remaining = budget - a
            for ctx in contexts:
                codes = self._unpack_context(ctx)
                self._recurse(ctx, codes, transitions, remaining, sink)

    def _recurse(
        self,
        ctx: int,
        codes: list[int],
        transitions_left: int,
        budget_left: int,
        sink: LineSink,
    ) -> None:
        if transitions_left == 0:
            ep = self._model.ep_level(ctx)
            if ep == budget_left:
                self._emit(codes, sink)
            return

        tt = transitions_left - 1
        low = tt * self._min_cp + self._min_ep
        high = tt * self._max_cp + self._max_ep
        a = self._a
        drop_mod = self._drop_mod
        for level, code_list in self._cp_buckets(ctx):
            if level > budget_left:
                break  # buckets are ascending; nothing further fits
            remaining = budget_left - level
            if remaining < low or remaining > high:
                continue
            for code in code_list:
                new_ctx = (ctx % drop_mod) * a + code
                codes.append(code)
                self._recurse(new_ctx, codes, tt, remaining, sink)
                codes.pop()

    def _emit(self, codes: list[int], sink: LineSink) -> None:
        cb = self._code_bytes
        sink.write_line_bytes(b"".join(cb[c] for c in codes))
        self._emitted += 1
        if self._limit is not None and self._emitted >= self._limit:
            raise _StopStreaming

    # -- helpers -----------------------------------------------------------

    def _cp_buckets(self, ctx: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
        cached = self._cp_cache.get(ctx)
        if cached is not None:
            return cached
        a = self._a
        base = ctx * a
        cp_levels = self._model.cp_levels
        by_level: dict[int, list[int]] = {}
        for code in range(a):
            by_level.setdefault(cp_levels[base + code], []).append(code)
        result = tuple((lvl, tuple(codes)) for lvl, codes in sorted(by_level.items()))
        self._cp_cache[ctx] = result
        return result

    def _unpack_context(self, ctx: int) -> list[int]:
        codes = [0] * self._ctx_len
        a = self._a
        for i in range(self._ctx_len - 1, -1, -1):
            codes[i] = ctx % a
            ctx //= a
        return codes

    def _total_level_ceiling(self, lo: int, hi: int) -> int:
        model = self._model
        ceiling = 0
        for length in range(lo, hi + 1):
            transitions = length - self._ctx_len
            per_length = (
                model.ln_levels[length] + self._ip_max + transitions * self._max_cp + self._max_ep
            )
            ceiling = max(ceiling, per_length)
        return ceiling
