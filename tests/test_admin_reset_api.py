from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import src.api.admin_auth as admin_auth
    import src.api.visitor as visitor
    import src.config as config

    monkeypatch.setattr(config.settings, "admin_token", "envtok", raising=False)
    monkeypatch.setattr(
        config.settings, "chat_db_path", str(tmp_path / "chats.db"), raising=False
    )
    monkeypatch.setattr(config.settings, "qdrant_url", ":memory:", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(admin_auth, "PASSWORD_PATH", tmp_path / "admin.json")
    monkeypatch.setattr(visitor, "STATE_PATH", tmp_path / "visitors.json")
    monkeypatch.setattr(visitor, "VISITORS_DIR", tmp_path / "visitors")

    from src.api.app import app

    return TestClient(app), admin_auth


def test_reset_backdoor_uses_env_token(client):
    c, admin_auth = client
    # 模拟一个"设置了但已遗忘"的密码：env token 不能再登录
    admin_auth.set_password("forgotten")
    assert admin_auth.verify("forgotten") is True
    assert admin_auth.verify("envtok") is False

    # 后门：用 env ADMIN_TOKEN 重置
    res = c.post(
        "/api/admin/reset",
        json={"admin_token": "envtok", "new_password": "brandnew"},
    )
    assert res.status_code == 200
    assert admin_auth.verify("brandnew") is True
    assert admin_auth.verify("forgotten") is False


def test_reset_backdoor_rejects_wrong_token(client):
    c, admin_auth = client
    admin_auth.set_password("forgotten")
    res = c.post(
        "/api/admin/reset",
        json={"admin_token": "nope", "new_password": "brandnew"},
    )
    assert res.status_code == 400
    assert admin_auth.verify("forgotten") is True


def test_reset_backdoor_needs_env_token(client, monkeypatch):
    c, admin_auth = client
    import src.config as config

    monkeypatch.setattr(config.settings, "admin_token", "", raising=False)
    res = c.post(
        "/api/admin/reset",
        json={"admin_token": "anything", "new_password": "brandnew"},
    )
    assert res.status_code == 400
