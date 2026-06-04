"""Parity between the native C enumerator and the Python reference.

The C enumerator (`native/omen-enum`) must emit candidates in byte-for-byte the
same order as :class:`PyEnumerator`, so the two are interchangeable. These tests
are skipped when the binary has not been built (keeps CI green without a
compiler); build it with ``make -C native``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from omen.enumerate import PyEnumerator
from omen.model import NgramModel

REPO_ROOT = Path(__file__).resolve().parent.parent
OMEN_ENUM = REPO_ROOT / "native" / "omen-enum"

pytestmark = pytest.mark.skipif(
    not OMEN_ENUM.is_file(), reason="native/omen-enum not built (run `make -C native`)"
)


class _Recorder:
    def __init__(self) -> None:
        self.items: list[bytes] = []

    def write_line_bytes(self, raw: bytes) -> None:
        self.items.append(raw)


def _python_candidates(model: NgramModel, limit: int) -> list[bytes]:
    sink = _Recorder()
    PyEnumerator(model).stream(sink, max_guesses=limit)
    return sink.items


def _c_candidates(model_dir: Path, limit: int) -> list[bytes]:
    proc = subprocess.run(
        [str(OMEN_ENUM), str(model_dir), "--max-guesses", str(limit)],
        capture_output=True,
        timeout=120,
        check=True,
    )
    return proc.stdout.split(b"\n")[:-1]  # trailing newline -> empty final field


def test_parity_default(trained_model: NgramModel, tmp_path: Path) -> None:
    trained_model.save(tmp_path)
    limit = 50_000
    assert _c_candidates(tmp_path, limit) == _python_candidates(trained_model, limit)


def test_parity_with_length_filter(trained_model: NgramModel, tmp_path: Path) -> None:
    trained_model.save(tmp_path)
    # Length filter is applied identically on both sides via --min/--max-length.
    sink = _Recorder()
    PyEnumerator(trained_model).stream(sink, max_guesses=20_000, min_length=8, max_length=8)
    proc = subprocess.run(
        [
            str(OMEN_ENUM),
            str(tmp_path),
            "--max-guesses",
            "20000",
            "--min-length",
            "8",
            "--max-length",
            "8",
        ],
        capture_output=True,
        timeout=120,
        check=True,
    )
    c_items = proc.stdout.split(b"\n")[:-1]
    assert c_items == sink.items


def test_parity_full_tiny() -> None:
    """Full enumeration of a tiny model must match exactly (no truncation)."""
    import tempfile

    from omen.train import ModelTrainer, TrainingOptions

    model = ModelTrainer(TrainingOptions(ngram=2, alphabet="ab", levels=8, max_length=3)).train(
        lambda: iter(["a", "b", "ab", "ba", "aab", "bba", "abb"])
    )
    with tempfile.TemporaryDirectory() as d:
        model.save(d)
        c_items = _c_candidates(Path(d), 1_000)
    py_items = _python_candidates(model, 1_000)
    assert c_items == py_items
