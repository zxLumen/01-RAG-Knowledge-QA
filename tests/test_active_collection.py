from __future__ import annotations

from src import config
from src.vectorstore import active


def test_active_collection_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(active, "PATH", tmp_path / "active_collection.json")
    config.settings.qdrant_collection = "knowledge_base"

    active.save("col_x")
    assert active.load() == "col_x"

    config.settings.qdrant_collection = "knowledge_base"
    active.apply()
    assert config.settings.qdrant_collection == "col_x"


def test_active_collection_absent_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(active, "PATH", tmp_path / "missing.json")
    config.settings.qdrant_collection = "fallback"
    active.apply()
    assert config.settings.qdrant_collection == "fallback"
