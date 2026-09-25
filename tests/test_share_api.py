from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import src.config as config

    monkeypatch.setattr(config.settings, "demo_mode", True, raising=False)
    monkeypatch.setattr(
        config.settings, "chat_db_path", str(tmp_path / "chats.db"), raising=False
    )
    monkeypatch.setattr(config.settings, "qdrant_url", ":memory:", raising=False)
    # Relative data/ and qdrant_data/ paths resolve inside the temp dir.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "shared.md").write_text("# 共享\n内容", encoding="utf-8")
    (tmp_path / "data" / "own.md").write_text("# 私人", encoding="utf-8")

    import src.api.admin_auth as admin_auth
    import src.api.visitor as visitor
    from src.api.app import app

    monkeypatch.setattr(visitor, "STATE_PATH", tmp_path / "visitors.json")
    monkeypatch.setattr(visitor, "VISITORS_DIR", tmp_path / "visitors")
    monkeypatch.setattr(admin_auth, "password_set", lambda: True, raising=False)
    monkeypatch.setattr(
        admin_auth, "verify", lambda token: token == "tok", raising=False
    )
    return TestClient(app)


ADMIN = {"X-Admin-Token": "tok"}


def _visitor_cookie(client: TestClient) -> dict:
    res = client.get("/api/status")
    assert res.status_code == 200
    return {"rag_visitor": res.cookies.get("rag_visitor")}


def test_share_config_requires_admin(client):
    assert client.get("/api/share/config").status_code == 401
    assert client.get("/api/share/config", headers=ADMIN).status_code == 200


def test_shared_collections_visible_to_visitors(client):
    jar = _visitor_cookie(client)
    res = client.post(
        "/api/share/config",
        headers=ADMIN,
        json={"collections": ["共享库"], "paths": []},
    )
    assert res.status_code == 200
    assert res.json()["collections"] == ["共享库"]

    res = client.get("/api/collections", cookies=jar)
    assert res.status_code == 200
    body = res.json()
    assert body["collections"] == ["我的知识库", "共享库"]
    assert body["shared"] == ["共享库"]


def test_shared_paths_visible_and_readable_but_not_deletable(client):
    jar = _visitor_cookie(client)
    client.post(
        "/api/share/config",
        headers=ADMIN,
        json={"collections": [], "paths": ["data/shared.md"]},
    )

    res = client.get("/api/files", cookies=jar)
    assert res.status_code == 200
    body = res.json()
    assert "data/shared.md" in body["shared_paths"]
    shared_item = next(i for i in body["items"] if i["rel"] == "data/shared.md")
    assert shared_item["shared"] is True
    # the visitor's own file is not shared
    assert not any(i["rel"] == "data/own.md" for i in body["items"])

    # readable
    res = client.get("/api/files/content", cookies=jar, params={"rel": "data/shared.md"})
    assert res.status_code == 200
    assert "内容" in res.json()["content"]

    # not deletable by a visitor
    res = client.post("/api/files/delete", cookies=jar, json={"rels": ["data/shared.md"]})
    assert res.status_code == 200
    assert res.json()["deleted"] == []
    assert res.json()["skipped"][0]["rel"] == "data/shared.md"
