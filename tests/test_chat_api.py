from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import src.config as config

    monkeypatch.setattr(config.settings, "demo_mode", True, raising=False)
    # Chat store on temp db; qdrant in-memory so tests never touch the real
    # storage folder (the running server may hold its lock).
    monkeypatch.setattr(
        config.settings, "chat_db_path", str(tmp_path / "chats.db"), raising=False
    )
    monkeypatch.setattr(
        config.settings, "qdrant_url", ":memory:", raising=False
    )
    import src.api.admin_auth as admin_auth
    import src.api.visitor as visitor
    from src.api.app import app

    monkeypatch.setattr(visitor, "STATE_PATH", tmp_path / "visitors.json")
    monkeypatch.setattr(visitor, "VISITORS_DIR", tmp_path / "visitors")
    monkeypatch.setattr(
        admin_auth, "password_set", lambda: True, raising=False
    )
    monkeypatch.setattr(
        admin_auth, "verify", lambda token: token == "tok", raising=False
    )
    return TestClient(app)


def _visitor_cookie(client: TestClient) -> dict:
    res = client.get("/api/status")
    assert res.status_code == 200
    return {"rag_visitor": res.cookies.get("rag_visitor")}


def test_chat_sessions_reports_owner(client):
    jar = _visitor_cookie(client)
    client.post(
        "/api/chat/sync",
        cookies=jar,
        json={"session_id": "s1", "title": "t", "messages": []},
    )
    res = client.get("/api/chat/sessions", cookies=jar)
    assert res.status_code == 200
    assert res.json()["owner"] == "visitor_" + jar["rag_visitor"]

    # admin gets its own owner namespace, not the visitor's
    res = client.get("/api/chat/sessions", headers={"X-Admin-Token": "tok"})
    assert res.status_code == 200
    assert res.json()["owner"] == "admin"
    assert res.json()["sessions"] == []


def test_chat_sync_list_soft_delete_restore_roundtrip(client):
    jar = _visitor_cookie(client)
    owner = "visitor_" + jar["rag_visitor"]

    res = client.post(
        "/api/chat/sync",
        cookies=jar,
        json={
            "session_id": "sess-1",
            "title": "我的会话",
            "collection": owner,
            "messages": [{"role": "user", "content": "你好"}],
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["owner"] == owner
    assert body["deleted"] is False

    res = client.get("/api/chat/sessions", cookies=jar)
    assert res.status_code == 200
    assert [s["id"] for s in res.json()["sessions"]] == ["sess-1"]

    # admin sees it across owners
    res = client.get(
        "/api/chat/admin/list", headers={"X-Admin-Token": "tok"}
    )
    assert res.status_code == 200
    admin = res.json()["sessions"]
    assert len(admin) == 1
    rowid = admin[0]["rowid"]

    # soft delete keeps content on server
    res = client.post(
        "/api/chat/admin/delete",
        headers={"X-Admin-Token": "tok"},
        json={"rowid": rowid},
    )
    assert res.status_code == 200
    assert res.json()["deleted"] is True

    # visitor list no longer hides it
    assert client.get("/api/chat/sessions", cookies=jar).json()["sessions"] == []

    # restore brings it back for the owner
    res = client.post(
        "/api/chat/admin/restore",
        headers={"X-Admin-Token": "tok"},
        json={"rowid": rowid},
    )
    assert res.status_code == 200
    assert res.json()["deleted"] is False
    restored = client.get("/api/chat/sessions", cookies=jar).json()
    assert [s["id"] for s in restored["sessions"]] == ["sess-1"]

    # hard delete removes it everywhere
    res = client.post(
        "/api/chat/admin/hard-delete",
        headers={"X-Admin-Token": "tok"},
        json={"rowid": rowid},
    )
    assert res.status_code == 200
    assert client.get("/api/chat/admin/list", headers={"X-Admin-Token": "tok"}).json()["count"] == 0


def test_chat_admin_requires_token(client):
    jar = _visitor_cookie(client)
    client.post(
        "/api/chat/sync",
        cookies=jar,
        json={"session_id": "s", "title": "t", "messages": []},
    )
    res = client.get("/api/chat/admin/list")
    assert res.status_code == 401


def test_chat_sync_oversized_rejected(client):
    jar = _visitor_cookie(client)
    import src.chats.store as chat_store

    messages = [{"role": "user", "content": "x" * 1000}] * (chat_store.MAX_MESSAGES + 5)
    res = client.post(
        "/api/chat/sync",
        cookies=jar,
        json={"session_id": "big", "title": "t", "messages": messages},
    )
    assert res.status_code == 200
    assert len(res.json()["messages"]) == chat_store.MAX_MESSAGES
