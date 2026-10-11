from __future__ import annotations

import pytest

from src.retrieval import config as retrieval_config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval_config, "CONFIG_PATH", tmp_path / "retrieval_config.json")
    monkeypatch.setattr(retrieval_config, "_cache", None)
    yield retrieval_config
    monkeypatch.setattr(retrieval_config, "_cache", None)


def test_defaults_include_hybrid_and_chunk(cfg):
    data = cfg.load()
    assert data["hybrid_enabled"] is True
    assert data["rrf_k"] >= 1
    assert isinstance(data["chunk_size"], int)
    assert data["chunk_overlap"] < data["chunk_size"]


def test_save_clamps_and_persists(cfg):
    out = cfg.save(
        {
            "top_k": 9999,
            "dense_weight": -5,
            "hybrid_enabled": "off",
            "chunk_size": 300,
            "chunk_overlap": 500,
        }
    )
    assert out["top_k"] == 100
    assert out["dense_weight"] == 0.0
    assert out["hybrid_enabled"] is False
    assert out["chunk_overlap"] < out["chunk_size"]

    cfg._cache = None  # force a reload from disk
    again = cfg.load()
    assert again["top_k"] == 100
    assert again["hybrid_enabled"] is False
    assert cfg.CONFIG_PATH.exists()


def test_partial_update_keeps_other_fields(cfg):
    before = cfg.load()
    cfg.save({"top_k": 7})
    after = cfg.load()
    assert after["top_k"] == 7
    assert after["chunks_per_file"] == before["chunks_per_file"]


def test_none_values_are_ignored(cfg):
    cfg.save({"top_k": 9})
    cfg.save({"top_k": None, "rrf_k": None})
    assert cfg.load()["top_k"] == 9


def test_chunk_settings_reflect_saved_values(cfg):
    cfg.save({"chunk_size": 800, "chunk_overlap": 100})
    assert cfg.chunk_settings() == (800, 100)
