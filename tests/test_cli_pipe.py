"""End-to-end CLI tests, including the real pipe / SIGPIPE contract.

These mirror how ashfall drives the generator: launch ``omen generate`` as a
subprocess and consume its stdout through a pipe that may close early.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_passwords.txt"


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from omen.cli import main

    out = tmp_path_factory.mktemp("model")
    rc = main(["train", "-i", str(SAMPLE), "-m", str(out), "--max-length", "16"])
    assert rc == 0
    assert (out / "config.json").is_file()
    return out


def test_generate_max_guesses_count(model_dir: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "omen", "generate", "-m", str(model_dir), "--max-guesses", "1234"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0
    lines = proc.stdout.decode("utf-8").splitlines()
    assert len(lines) == 1234


def test_generate_pipe_into_head_is_clean(model_dir: Path) -> None:
    """`omen generate | head -5` yields 5 lines and no traceback (SIGPIPE clean)."""
    gen = f"{shlex.quote(sys.executable)} -m omen generate -m {shlex.quote(str(model_dir))}"
    pipeline = f"{gen} | head -n 5"
    proc = subprocess.run(
        ["bash", "-c", pipeline],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        timeout=120,
    )
    out_lines = proc.stdout.decode("utf-8").splitlines()
    assert len(out_lines) == 5
    stderr = proc.stderr.decode("utf-8")
    assert "Traceback" not in stderr
    assert "BrokenPipe" not in stderr


def test_eval_cli(model_dir: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "omen", "eval", "-m", str(model_dir), "password", "123456"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0
    out = proc.stdout.decode("utf-8")
    assert "password\tlevel=" in out
    assert "123456\tlevel=" in out


def test_inspect_cli(model_dir: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "omen", "inspect", "-m", str(model_dir)],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "OMEN model" in proc.stdout.decode("utf-8")


def test_alphabet_cli() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "omen", "alphabet", "-i", str(SAMPLE), "--size", "20"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "coverage" in proc.stdout.decode("utf-8")
