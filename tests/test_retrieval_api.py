from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.retrieval import config as retrieval_config
from src.retrieval.hybrid import RetrievedChunk


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import src.api.admin_auth as admin_auth
    import src.config as config
    from src.api import routes
    from src.api.app import app

    monkeypatch.setattr(config.settings, "demo_mode", False, raising=False)
    monkeypatch.setattr(config.settings, "qdrant_url", ":memory:", raising=False)
    monkeypatch.setattr(
        retrieval_config, "CONFIG_PATH", tmp_path / "retrieval_config.json"
    )
    monkeypatch.setattr(retrieval_config, "_cache", None)
    monkeypatch.setattr(admin_auth, "password_set", lambda: True, raising=False)
    monkeypatch.setattr(
        admin_auth, "verify", lambda token: token == "tok", raising=False
    )

    def fake_search(question, top_k=None, collection_name=None, chunks_per_file=None):
        return [
            RetrievedChunk(
                text="hello world",
                score=0.5,
                metadata={"filename": "a.md", "rel_path": "data/a.md"},
            )
        ]

    monkeypatch.setattr(routes, "retrieval_search", fake_search)
    yield TestClient(app)
    monkeypatch.setattr(retrieval_config, "_cache", None)


def test_config_exposes_retrieval_fields(client):
    data = client.get("/api/config").json()
    assert data["chunks_per_file"] >= 1
    assert data["hybrid_enabled"] in (True, False)
    assert "rrf_k" in data


def test_retrieval_config_read_and_write(client):
    res = client.get("/api/retrieval/config")
    assert res.status_code == 200
    assert res.json()["hybrid_enabled"] in (True, False)

    # Writing requires the admin token.
    assert client.post("/api/retrieval/config", json={"top_k": 7}).status_code == 401

    res = client.post(
        "/api/retrieval/config",
        json={"top_k": 7, "hybrid_enabled": False},
        headers={"X-Admin-Token": "tok"},
    )
    assert res.status_code == 200
    assert res.json()["top_k"] == 7
    assert res.json()["hybrid_enabled"] is False

    again = client.get("/api/retrieval/config").json()
    assert again["top_k"] == 7
    assert again["hybrid_enabled"] is False


def test_retrieve_returns_chunks_and_sources(client):
    res = client.post("/api/retrieve", json={"question": "hi", "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 1
    assert data["sources"] == ["a.md"]
    assert data["chunks"][0]["text"] == "hello world"
    assert data["chunks"][0]["rel_path"] == "data/a.md"
