"""Streaming I/O helpers.

Two responsibilities, deliberately small:

* :class:`ByteSink` — a buffered, newline-terminating writer over a binary
  stream.  Candidate generation produces millions of short lines; writing each
  one individually to ``stdout`` is dominated by per-call overhead, so we batch
  into a ``bytearray`` and flush in large chunks.
* :func:`read_corpus` — a memory-bounded line iterator over a training corpus,
  so we never load an arbitrarily large password file fully into memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

# Flush threshold for the candidate stream.  ~1 MiB balances syscall overhead
# against memory and latency to the downstream consumer.
_FLUSH_BYTES = 1 << 20


@runtime_checkable
class LineSink(Protocol):
    """Anything the enumerator can write candidate lines into.

    Satisfied by :class:`ByteSink` (real output) and by counting/null sinks used
    for scoring and tests, so the enumerator does not care where lines go.
    """

    def write_line_bytes(self, raw: bytes) -> None:
        """Consume one already-encoded candidate (without a trailing newline)."""
        ...


class ByteSink:
    """Buffered newline-terminating writer over a binary stream.

    The sink appends ``\\n`` after every line and flushes whenever the internal
    buffer crosses :data:`_FLUSH_BYTES`.  Call :meth:`close` (or use it as a
    context manager) to flush the tail.

    A broken downstream pipe (the consumer exited) surfaces as
    :class:`BrokenPipeError` from :meth:`write_line`/:meth:`flush`; callers are
    expected to treat that as a normal end-of-consumer condition.
    """

    __slots__ = ("_buf", "_stream")

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._buf = bytearray()

    def write_line(self, text: str) -> None:
        """Encode ``text`` as UTF-8, append a newline, and buffer it."""
        self._buf += text.encode("utf-8")
        self._buf.append(0x0A)
        if len(self._buf) >= _FLUSH_BYTES:
            self.flush()

    def write_line_bytes(self, raw: bytes) -> None:
        """Buffer already-encoded ``raw`` bytes followed by a newline."""
        self._buf += raw
        self._buf.append(0x0A)
        if len(self._buf) >= _FLUSH_BYTES:
            self.flush()

    def flush(self) -> None:
        """Write the buffer to the underlying stream and clear it."""
        if self._buf:
            self._stream.write(self._buf)
            self._buf.clear()
        self._stream.flush()

    def close(self) -> None:
        """Flush any buffered tail.  Does not close the wrapped stream."""
        self.flush()

    def __enter__(self) -> ByteSink:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@contextmanager
def read_corpus(path: Path | str) -> Iterator[Iterator[str]]:
    """Yield an iterator of passwords from ``path`` (a file or ``-`` for stdin).

    Lines are decoded as UTF-8 with ``errors="replace"`` (corpora are untrusted
    and may contain invalid byte sequences).  Only the trailing ``\\n``/``\\r``
    is stripped — interior whitespace is part of the password.  Empty lines are
    skipped.  Reading is lazy, so the file is never fully materialised.
    """
    if str(path) == "-":
        import sys

        yield _iter_lines(sys.stdin.buffer.read().decode("utf-8", "replace").splitlines())
        return

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        yield _iter_lines(handle)


def _iter_lines(lines: Iterator[str] | list[str]) -> Iterator[str]:
    for line in lines:
        pw = line.rstrip("\r\n")
        if pw:
            yield pw
