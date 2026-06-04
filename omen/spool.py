"""Chunked spool-and-attack — feed a candidate stream to hashcat at full speed.

Piping a generator straight into ``hashcat`` over stdin caps throughput: hashcat
cannot ``mmap`` a pipe, loses its wordlist amplifier, and stalls on producer
backpressure (the classic "fast burst, then ~300 MH/s" behaviour). A named pipe
(FIFO) doesn't help — hashcat ``mmap``s its wordlist argument and a FIFO is not
seekable/sizable, so the map fails.

The fix is a **RAM-backed file** that hashcat *can* ``mmap``: spool the producer's
output into bounded chunk files on tmpfs (``/dev/shm``), and attack each chunk
with ``hashcat -a 0 <chunk> …``. Chunks are **double-buffered** — the next chunk
is filled while the current one is being attacked — so generation latency hides
behind GPU work and the GPU stays fed.
"""

from __future__ import annotations

import contextlib
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

# hashcat exit codes we care about (per-chunk semantics):
#   0 = all hashes cracked  -> stop the whole run
#   1 = exhausted this chunk -> move on to the next chunk
#  >=2 = aborted/error       -> stop
HC_CRACKED = 0
HC_EXHAUSTED = 1

CHUNK_PLACEHOLDER = "{chunk}"
_DEFAULT_READ_BLOCK = 1 << 22  # 4 MiB reads from the producer
_DEFAULT_CHUNK_BYTES = 512 << 20  # 512 MiB per chunk


@dataclass(frozen=True, slots=True)
class SpoolConfig:
    """Tunables for chunked spooling."""

    chunk_bytes: int = _DEFAULT_CHUNK_BYTES
    chunk_lines: int = 0  # 0 = no per-chunk line cap (bytes only)
    max_guesses: int = 0  # 0 = unlimited
    spool_dir: str | None = None  # None = auto (/dev/shm, else temp dir)
    read_block: int = _DEFAULT_READ_BLOCK

    def resolve_spool_dir(self) -> Path:
        """Pick a RAM-backed spool directory, falling back to the temp dir."""
        if self.spool_dir is not None:
            return Path(self.spool_dir)
        shm = Path("/dev/shm")
        if shm.is_dir():
            return shm
        return Path(tempfile.gettempdir())


@dataclass(frozen=True, slots=True)
class SpoolResult:
    """Outcome of a spool run."""

    chunks: int
    candidates: int
    status: str  # "exhausted" | "cracked" | "aborted"


def _index_after_nth_newline(data: bytes, n: int) -> int:
    """Byte index just past the ``n``-th newline in ``data`` (len if fewer)."""
    idx = -1
    for _ in range(n):
        nxt = data.find(b"\n", idx + 1)
        if nxt == -1:
            return len(data)
        idx = nxt
    return idx + 1


class ChunkSpooler:
    """Streams a producer's stdout into newline-aligned tmpfs chunk files.

    Use :meth:`chunks` as an iterator of ready chunk paths. A background thread
    fills the *next* chunk while the caller consumes the current one (one chunk
    of look-ahead = double buffering). The caller owns each yielded file and
    should delete it once consumed; any not consumed are cleaned up on exit.
    """

    def __init__(self, producer_argv: Sequence[str], config: SpoolConfig) -> None:
        if not producer_argv:
            raise ValueError("producer_argv must not be empty")
        self._argv = list(producer_argv)
        self._cfg = config
        self._dir = config.resolve_spool_dir()
        self._stop = threading.Event()
        self._proc: subprocess.Popen[bytes] | None = None
        self._error: BaseException | None = None
        self.candidates = 0

    # -- context management ------------------------------------------------

    def __enter__(self) -> ChunkSpooler:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._terminate_producer()

    def request_stop(self) -> None:
        """Ask the filler to stop early (e.g. all hashes cracked)."""
        self._stop.set()

    # -- public iteration --------------------------------------------------

    def chunks(self) -> Iterator[Path]:
        """Yield ready chunk file paths in order, double-buffered."""
        ready: queue.Queue[Path | None] = queue.Queue(maxsize=1)
        filler = threading.Thread(target=self._fill, args=(ready,), daemon=True)
        filler.start()
        try:
            while True:
                item = ready.get()
                if item is None:
                    break
                yield item
        finally:
            self._stop.set()
            self._drain(ready, filler)
            filler.join(timeout=5.0)
            if self._error is not None:
                raise self._error

    # -- internals ---------------------------------------------------------

    def _drain(self, ready: queue.Queue[Path | None], filler: threading.Thread) -> None:
        """Unblock and empty the queue so a filler stuck on put() can exit."""
        while True:
            try:
                item = ready.get(timeout=0.1)
            except queue.Empty:
                if not filler.is_alive():
                    return
                continue
            if item is None:
                return
            item.unlink(missing_ok=True)

    def _safe_put(self, ready: queue.Queue[Path | None], path: Path) -> bool:
        """Enqueue a chunk, honouring an early stop request. False = dropped."""
        while not self._stop.is_set():
            try:
                ready.put(path, timeout=0.1)
                return True
            except queue.Full:
                continue
        path.unlink(missing_ok=True)
        return False

    def _open_chunk(self) -> tuple[Path, BinaryIO]:
        self._dir.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self._dir, prefix="omen_chunk_", suffix=".txt")
        return Path(name), os.fdopen(fd, "wb")

    def _terminate_producer(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()

    def _fill(self, ready: queue.Queue[Path | None]) -> None:
        cfg = self._cfg
        try:
            self._proc = subprocess.Popen(
                self._argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            stream = self._proc.stdout
            assert stream is not None

            carry = b""
            path: Path | None = None
            handle: BinaryIO | None = None
            cur_bytes = 0
            cur_lines = 0
            reached_max = False
            eof = False

            while not self._stop.is_set() and not reached_max:
                block = stream.read(cfg.read_block)
                if not block:
                    eof = True
                    break
                data = carry + block
                nl = data.rfind(b"\n")
                if nl == -1:
                    carry = data
                    continue
                to_write = data[: nl + 1]
                carry = data[nl + 1 :]

                if cfg.max_guesses:
                    incoming = to_write.count(b"\n")
                    if self.candidates + incoming >= cfg.max_guesses:
                        cut = _index_after_nth_newline(to_write, cfg.max_guesses - self.candidates)
                        to_write = to_write[:cut]
                        reached_max = True

                if handle is None:
                    path, handle = self._open_chunk()
                    cur_bytes = cur_lines = 0
                handle.write(to_write)
                added = to_write.count(b"\n")
                cur_bytes += len(to_write)
                cur_lines += added
                self.candidates += added

                roll = cur_bytes >= cfg.chunk_bytes or (
                    cfg.chunk_lines and cur_lines >= cfg.chunk_lines
                )
                if roll and path is not None:
                    handle.close()
                    self._safe_put(ready, path)
                    path, handle = None, None

            # Flush the final partial chunk. A carried, unterminated final line
            # is a real candidate only at genuine EOF — not when we stopped at
            # the guess cap or on request (its line lies beyond the budget).
            if handle is not None and path is not None:
                if eof and carry:
                    handle.write(carry + b"\n")
                    self.candidates += 1
                handle.close()
                self._safe_put(ready, path)
        except BaseException as exc:
            # Any failure is surfaced to the consumer when it joins the thread.
            self._error = exc
        finally:
            self._terminate_producer()
            with contextlib.suppress(queue.Full):
                ready.put(None, timeout=1.0)


class HashcatRunner:
    """Runs hashcat once per chunk, substituting ``{chunk}`` in the template."""

    def __init__(self, template: Sequence[str]) -> None:
        if CHUNK_PLACEHOLDER not in template:
            raise ValueError(f"hashcat template must contain {CHUNK_PLACEHOLDER!r}")
        self._template = list(template)

    def attack(self, chunk: Path) -> int:
        """Attack one chunk file; return hashcat's exit code."""
        cmd = [chunk.as_posix() if arg == CHUNK_PLACEHOLDER else arg for arg in self._template]
        return subprocess.run(cmd, check=False).returncode


def run_spool(
    producer_argv: Sequence[str],
    hashcat_template: Sequence[str],
    config: SpoolConfig,
) -> SpoolResult:
    """Spool ``producer_argv`` output into chunks and attack each with hashcat.

    Stops when the producer is exhausted, the guess budget is hit, or hashcat
    reports all hashes cracked (exit code 0).
    """
    runner = HashcatRunner(hashcat_template)
    status = "exhausted"
    n_chunks = 0
    with ChunkSpooler(producer_argv, config) as spooler:
        for chunk in spooler.chunks():
            n_chunks += 1
            rc = runner.attack(chunk)
            chunk.unlink(missing_ok=True)
            if rc == HC_CRACKED:
                status = "cracked"
                spooler.request_stop()
                break
            if rc != HC_EXHAUSTED:
                status = "aborted"
                spooler.request_stop()
                break
        candidates = spooler.candidates
    return SpoolResult(chunks=n_chunks, candidates=candidates, status=status)


def have_hashcat() -> bool:
    """Whether a ``hashcat`` binary is on PATH (for CLI diagnostics)."""
    return shutil.which("hashcat") is not None
