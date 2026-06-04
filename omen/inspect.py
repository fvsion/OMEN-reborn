"""Human-readable model introspection.

:class:`ModelInspector` renders a model's configuration, table sizes, alphabet
coverage, and per-table level histograms as text — useful for sanity-checking a
freshly trained model before generating from it.
"""

from __future__ import annotations

from collections.abc import Iterable

from omen.model import NgramModel


class ModelInspector:
    """Produces a textual summary of a trained :class:`NgramModel`."""

    def __init__(self, model: NgramModel) -> None:
        self._model = model

    def report(self) -> str:
        """Return a multi-line, human-readable description of the model."""
        m = self._model
        lines: list[str] = []
        lines.append("OMEN model")
        lines.append("=" * 60)
        lines.append(f"  ngram (n)        : {m.ngram}")
        lines.append(f"  context length   : {m.context_len}")
        lines.append(f"  levels (NL)      : {m.scale.levels}")
        lines.append(f"  lam (nats/level) : {m.scale.lam:.6g}")
        lines.append(f"  smoothing (δ)    : {m.smoothing:.6g}")
        lines.append(f"  length range     : [{m.min_length}, {m.max_length}]")
        lines.append(f"  EP enabled       : {m.ep_enabled}")
        lines.append(f"  alphabet size    : {m.alphabet_size}")
        lines.append(f"  alphabet coverage: {m.coverage:.2%}")
        lines.append(f"  alphabet         : {self._render_alphabet()}")
        lines.append("")
        lines.append("Table sizes (entries)")
        lines.append("-" * 60)
        lines.append(f"  IP : {len(m.ip_levels):>12,}")
        lines.append(f"  CP : {len(m.cp_levels):>12,}")
        lines.append(f"  EP : {len(m.ep_levels):>12,}")
        lines.append(f"  LN : {len(m.ln_levels):>12,}")
        lines.append("")
        lines.append("Level histograms (entries per level)")
        lines.append("-" * 60)
        lines.append(self._histogram_block("IP", m.ip_levels))
        lines.append(self._histogram_block("CP", m.cp_levels))
        if m.ep_enabled:
            lines.append(self._histogram_block("EP", m.ep_levels))
        lines.append(self._histogram_block("LN", m.ln_levels[m.min_length :]))
        return "\n".join(lines)

    def _render_alphabet(self) -> str:
        # Escape control/whitespace so the report stays single-line and readable.
        rendered = []
        for ch in self._model.alphabet.chars:
            if ch.isprintable() and ch != " ":
                rendered.append(ch)
            else:
                rendered.append(repr(ch).strip("'"))
        return "".join(rendered)

    def _histogram_block(self, name: str, data: Iterable[int]) -> str:
        counts = self._histogram(data)
        total = sum(counts)
        width = 40
        rows = [f"  {name}"]
        peak = max(counts) if counts else 0
        for level, count in enumerate(counts):
            if count == 0:
                continue
            bar = "#" * round(width * count / peak) if peak else ""
            pct = (count / total) if total else 0.0
            rows.append(f"    L{level:>3} | {count:>10,} {pct:6.1%} {bar}")
        return "\n".join(rows)

    def _histogram(self, data: Iterable[int]) -> list[int]:
        counts = [0] * self._model.scale.levels
        for level in data:
            if 0 <= level < len(counts):
                counts[level] += 1
        return counts
