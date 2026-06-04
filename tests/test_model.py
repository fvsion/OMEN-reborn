"""Tests for the model: training, query, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omen.errors import ConfigError, ModelError, TrainingError
from omen.model import (
    _MANIFEST_HEADER,
    CONFIG_NAME,
    MANIFEST_MAGIC,
    MANIFEST_NAME,
    NgramModel,
)
from omen.train import ModelTrainer, TrainingOptions


def test_training_produces_consistent_tables(trained_model: NgramModel) -> None:
    m = trained_model
    assert m.ngram == 3
    assert m.context_len == 2
    assert len(m.ip_levels) == m.alphabet_size**2
    assert len(m.cp_levels) == m.alphabet_size**3
    assert len(m.ep_levels) == m.alphabet_size**2
    assert len(m.ln_levels) == m.max_length + 1
    assert 0.0 < m.coverage <= 1.0


def test_save_load_roundtrip(trained_model: NgramModel, tmp_path: Path) -> None:
    trained_model.save(tmp_path)
    loaded = NgramModel.load(tmp_path)
    assert loaded.ngram == trained_model.ngram
    assert loaded.alphabet.as_string() == trained_model.alphabet.as_string()
    assert loaded.scale.levels == trained_model.scale.levels
    assert loaded.scale.lam == pytest.approx(trained_model.scale.lam)
    assert loaded.ip_levels == trained_model.ip_levels
    assert loaded.cp_levels == trained_model.cp_levels
    assert loaded.ep_levels == trained_model.ep_levels
    assert loaded.ln_levels == trained_model.ln_levels


def test_total_level_matches_components(trained_model: NgramModel) -> None:
    codes = trained_model.alphabet.encode("password")
    assert codes is not None
    total = trained_model.total_level(codes)
    assert total is not None and total >= 0


def test_total_level_none_for_out_of_range(trained_model: NgramModel) -> None:
    too_short = trained_model.alphabet.encode("a")  # shorter than context length
    assert too_short is not None
    assert trained_model.total_level(too_short) is None


def test_explicit_alphabet_skips_foreign_passwords() -> None:
    options = TrainingOptions(ngram=2, alphabet="abc", max_length=8)
    model = ModelTrainer(options).train(lambda: iter(["abc", "abc", "xyz", "cab"]))
    assert model.alphabet.as_string() == "abc"


def test_training_empty_corpus_raises() -> None:
    options = TrainingOptions(ngram=3, alphabet="abc")
    with pytest.raises(TrainingError):
        ModelTrainer(options).train(lambda: iter([]))


def test_manifest_roundtrips(trained_model: NgramModel, tmp_path: Path) -> None:
    """manifest.bin must faithfully encode the fields the C enumerator needs."""
    trained_model.save(tmp_path)
    raw = (tmp_path / MANIFEST_NAME).read_bytes()
    (
        magic,
        _version,
        ngram,
        levels,
        max_length,
        ep_enabled,
        alpha_size,
        ln_len,
        alpha_bytes,
        lam,
    ) = _MANIFEST_HEADER.unpack_from(raw, 0)
    assert magic == MANIFEST_MAGIC
    assert ngram == trained_model.ngram
    assert levels == trained_model.scale.levels
    assert max_length == trained_model.max_length
    assert ep_enabled == int(trained_model.ep_enabled)
    assert alpha_size == trained_model.alphabet_size
    assert ln_len == len(trained_model.ln_levels)
    assert lam == pytest.approx(trained_model.scale.lam)

    off = _MANIFEST_HEADER.size
    code_lens = raw[off : off + alpha_size]
    off += alpha_size
    alpha_utf8 = raw[off : off + alpha_bytes]
    off += alpha_bytes
    ln_levels = raw[off : off + ln_len]
    # Reconstruct the alphabet from per-code UTF-8 lengths.
    chars, pos = [], 0
    for clen in code_lens:
        chars.append(alpha_utf8[pos : pos + clen].decode("utf-8"))
        pos += clen
    assert "".join(chars) == trained_model.alphabet.as_string()
    assert tuple(ln_levels) == trained_model.ln_levels
    assert off + ln_len == len(raw)  # no trailing junk


def test_load_rejects_missing_config(tmp_path: Path) -> None:
    with pytest.raises(ModelError):
        NgramModel.load(tmp_path)


def test_load_rejects_tampered_ngram(trained_model: NgramModel, tmp_path: Path) -> None:
    trained_model.save(tmp_path)
    config_path = tmp_path / CONFIG_NAME
    config = json.loads(config_path.read_text())
    config["ngram"] = 99  # exceeds MAX_NGRAM
    config_path.write_text(json.dumps(config))
    with pytest.raises(ConfigError):
        NgramModel.load(tmp_path)


def test_load_rejects_truncated_table(trained_model: NgramModel, tmp_path: Path) -> None:
    trained_model.save(tmp_path)
    (tmp_path / "cp.dat").write_bytes(b"\x00\x01\x02")  # wrong size
    with pytest.raises(ConfigError):
        NgramModel.load(tmp_path)
