#!/usr/bin/env python3
"""Guessing benchmark: cumulative held-out crack rate vs. number of guesses.

Each *method* produces an ordered stream of candidate passwords. We feed that
stream through a :class:`CrackCounter` that tracks how many unique held-out test
passwords have been matched at a set of (log-spaced) guess milestones, and write
the curve into a shared ``results.json``.

Methods are run one at a time and merged into ``results.json`` so an expensive
run (OMEN) need not be repeated when adding another (PRINCE):

    python run_benchmark.py omen     --model /tmp/ry_model  --label "OMEN n=3"
    python run_benchmark.py omen     --model /tmp/ry_model4 --label "OMEN n=4"
    python run_benchmark.py wordlist --wordlist /tmp/ry_train.txt --label "rockyou wordlist"
    python run_benchmark.py prince   --prince-bin ./pp64.bin --wordlist /tmp/ry_words.txt

The test set, max-guesses, and sample grid are fixed across methods (stored in
``results.json`` meta and reused), so all curves are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results.json"
DEFAULT_MAX_GUESSES = 20_000_000


def log_grid(start: int, stop: int, per_decade: int = 20) -> list[int]:
    """Integer, strictly increasing log-spaced sample points in [start, stop]."""
    points: list[int] = []
    value = float(start)
    factor = 10.0 ** (1.0 / per_decade)
    while value <= stop:
        v = round(value)
        if not points or v > points[-1]:
            points.append(v)
        value *= factor
    if points[-1] != stop:
        points.append(stop)
    return points


class CrackCounter:
    """Counts unique held-out cracks at predefined guess milestones."""

    def __init__(self, test: set[bytes], samples: list[int], max_guesses: int) -> None:
        self._test = test
        self._samples = samples
        self.max_guesses = max_guesses
        self.guesses = 0
        self.cracked = 0
        self._idx = 0
        self.out_guesses: list[int] = []
        self.out_cracked: list[int] = []

    def feed(self, candidate: bytes) -> bool:
        """Record one guess. Returns False once the guess budget is exhausted."""
        self.guesses += 1
        if candidate in self._test:
            self._test.discard(candidate)
            self.cracked += 1
        while self._idx < len(self._samples) and self.guesses >= self._samples[self._idx]:
            self.out_guesses.append(self.guesses)
            self.out_cracked.append(self.cracked)
            self._idx += 1
        return self.guesses < self.max_guesses

    def finalize(self) -> None:
        """Record a final point if the stream ended before the last milestone."""
        if not self.out_guesses or self.out_guesses[-1] != self.guesses:
            self.out_guesses.append(self.guesses)
            self.out_cracked.append(self.cracked)


def load_test_set(path: Path) -> set[bytes]:
    test: set[bytes] = set()
    with path.open("rb") as handle:
        for line in handle:
            pw = line.rstrip(b"\n")
            if pw:
                test.add(pw)
    return test


# -- candidate sources -----------------------------------------------------


class _Done(Exception):
    """Raised to unwind the OMEN stream once the guess budget is reached."""


def run_omen(model_dir: str, counter: CrackCounter) -> None:
    import contextlib

    from omen.enumerate import PyEnumerator
    from omen.model import NgramModel

    class _Sink:
        __slots__ = ()

        def write_line_bytes(self, raw: bytes) -> None:
            if not counter.feed(raw):
                raise _Done

    model = NgramModel.load(model_dir)
    with contextlib.suppress(_Done):
        PyEnumerator(model).stream(_Sink(), max_guesses=counter.max_guesses)


def run_stream(lines: Iterator[bytes], counter: CrackCounter) -> None:
    for raw in lines:
        if not counter.feed(raw.rstrip(b"\n")):
            break


def wordlist_lines(path: str) -> Iterator[bytes]:
    with open(path, "rb") as handle:
        yield from handle


def run_prince(prince_bin: str, wordlist: str, counter: CrackCounter) -> None:
    proc = subprocess.Popen(
        [prince_bin, wordlist],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    try:
        run_stream(iter(proc.stdout.readline, b""), counter)
    finally:
        proc.kill()
        proc.wait()


# -- results persistence ---------------------------------------------------


def load_results(path: Path, test_path: Path, max_guesses: int) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    test_unique = len(load_test_set(test_path))
    samples = log_grid(1_000, max_guesses, per_decade=20)
    return {
        "meta": {
            "dataset": "rockyou",
            "split": "md5(pw) % 10 == 0 -> held-out test (disjoint from train)",
            "test_path": str(test_path),
            "test_unique": test_unique,
            "max_guesses": max_guesses,
            "samples": samples,
        },
        "methods": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("method", choices=["omen", "wordlist", "prince"])
    parser.add_argument("--label", help="curve label (defaults per method)")
    parser.add_argument("--model", help="OMEN model directory")
    parser.add_argument("--wordlist", help="wordlist path (wordlist/prince)")
    parser.add_argument("--prince-bin", help="path to pp64 binary")
    parser.add_argument("--test", default="/tmp/ry_test.txt", help="held-out test file")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--max-guesses", type=int, default=DEFAULT_MAX_GUESSES)
    args = parser.parse_args()

    results_path = Path(args.results)
    test_path = Path(args.test)
    results = load_results(results_path, test_path, args.max_guesses)
    max_guesses = results["meta"]["max_guesses"]
    samples = results["meta"]["samples"]

    test = load_test_set(test_path)
    counter = CrackCounter(test, samples, max_guesses)

    t0 = time.time()
    if args.method == "omen":
        label = args.label or "OMEN"
        run_omen(args.model, counter)
    elif args.method == "wordlist":
        label = args.label or "wordlist"
        run_stream(wordlist_lines(args.wordlist), counter)
    else:  # prince
        label = args.label or "PRINCE"
        run_prince(args.prince_bin, args.wordlist, counter)
    counter.finalize()
    elapsed = time.time() - t0

    total = results["meta"]["test_unique"]
    results["methods"][label] = {
        "guesses": counter.out_guesses,
        "cracked": counter.out_cracked,
        "final_crackrate": counter.cracked / total,
        "elapsed_s": round(elapsed, 1),
    }
    results_path.write_text(json.dumps(results, indent=2))
    print(
        f"{label}: {counter.cracked:,}/{total:,} = {counter.cracked / total:.2%} "
        f"in {counter.guesses:,} guesses ({elapsed:.0f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
