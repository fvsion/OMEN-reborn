"""Tests for the chunked spool-and-attack runner (no GPU required).

A fake producer emits ``candN`` lines; a fake "hashcat" just inspects each chunk
file. This exercises chunk rollover, newline-aligned boundaries, double-buffer
recycling, the guess cap, and the all-cracked early stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

from omen.spool import SpoolConfig, run_spool

# Fake hashcat: count lines in the chunk, count malformed (non-"cand") lines,
# append "<count> <bad>" to a log, and exit 1 (exhausted) — or 0 (cracked) if a
# target token is present.
_FAKE_HC = """
import sys
chunk, log = sys.argv[1], sys.argv[2]
target = sys.argv[3] if len(sys.argv) > 3 else None
lines = open(chunk, "rb").read().split(b"\\n")
if lines and lines[-1] == b"":
    lines.pop()
bad = sum(1 for ln in lines if not ln.startswith(b"cand"))
with open(log, "a") as f:
    f.write(f"{len(lines)} {bad}\\n")
sys.exit(0 if (target and target.encode() in lines) else 1)
"""


def _producer(n: int) -> list[str]:
    code = f"import sys\nw=sys.stdout.write\nfor i in range({n}): w('cand%d\\n' % i)"
    return [sys.executable, "-c", code]


def _fake_hc(tmp_path: Path) -> Path:
    path = tmp_path / "fake_hc.py"
    path.write_text(_FAKE_HC)
    return path


def _read_log(log: Path) -> tuple[int, int]:
    delivered = malformed = 0
    for line in log.read_text().splitlines():
        n, bad = line.split()
        delivered += int(n)
        malformed += int(bad)
    return delivered, malformed


def test_all_candidates_delivered_in_clean_chunks(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    hc = _fake_hc(tmp_path)
    template = [sys.executable, str(hc), "{chunk}", str(log)]
    # Tiny reads + chunks force many rollovers and cross-read partial lines.
    cfg = SpoolConfig(chunk_bytes=200, read_block=64, spool_dir=str(tmp_path))

    result = run_spool(_producer(1000), template, cfg)

    assert result.status == "exhausted"
    assert result.candidates == 1000
    assert result.chunks >= 2  # actually rolled over
    delivered, malformed = _read_log(log)
    assert delivered == 1000  # every candidate reached hashcat exactly once
    assert malformed == 0  # no candidate was split across a chunk boundary


def test_max_guesses_is_honoured(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    hc = _fake_hc(tmp_path)
    template = [sys.executable, str(hc), "{chunk}", str(log)]
    cfg = SpoolConfig(chunk_bytes=200, read_block=64, max_guesses=100, spool_dir=str(tmp_path))

    result = run_spool(_producer(1000), template, cfg)

    assert result.candidates == 100
    delivered, malformed = _read_log(log)
    assert delivered == 100
    assert malformed == 0


def test_stops_early_when_cracked(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    hc = _fake_hc(tmp_path)
    # "cand5" lands in the first chunk -> fake hc exits 0 -> stop immediately.
    template = [sys.executable, str(hc), "{chunk}", str(log), "cand5"]
    cfg = SpoolConfig(chunk_bytes=200, read_block=64, spool_dir=str(tmp_path))

    result = run_spool(_producer(100_000), template, cfg)

    assert result.status == "cracked"
    assert result.chunks == 1  # stopped after the first chunk
    # producer was terminated early, so far fewer than 100k candidates spooled
    assert result.candidates < 100_000


def test_chunk_files_are_cleaned_up(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    hc = _fake_hc(tmp_path)
    template = [sys.executable, str(hc), "{chunk}", str(log)]
    cfg = SpoolConfig(chunk_bytes=200, read_block=64, spool_dir=str(tmp_path))

    run_spool(_producer(500), template, cfg)

    leftover = list(tmp_path.glob("omen_chunk_*"))
    assert leftover == []  # every chunk file deleted after use
