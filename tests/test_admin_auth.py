from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def admin_auth(tmp_path, monkeypatch):
    import src.config as config

    monkeypatch.setattr(config.settings, "admin_token", "env-secret", raising=False)
    mod = importlib.import_module("src.api.admin_auth")
    monkeypatch.setattr(mod, "PASSWORD_PATH", tmp_path / "admin.json")
    return mod


def test_env_fallback_verify(admin_auth):
    assert admin_auth.verify("env-secret") is True
    assert admin_auth.verify("wrong") is False
    assert admin_auth.verify("") is False
    assert admin_auth.password_set() is True


def test_password_set_and_verify(admin_auth):
    admin_auth.set_password("newpass")
    assert admin_auth.verify("newpass") is True
    assert admin_auth.verify("env-secret") is False
    assert admin_auth.verify("wrong") is False


def test_change_password(admin_auth):
    admin_auth.set_password("oldpass")
    admin_auth.change_password("oldpass", "newpass")
    assert admin_auth.verify("newpass") is True
    assert admin_auth.verify("oldpass") is False


def test_change_password_wrong_old(admin_auth):
    admin_auth.set_password("oldpass")
    with pytest.raises(ValueError):
        admin_auth.change_password("bad", "newpass")
    assert admin_auth.verify("oldpass") is True


def test_set_password_too_short(admin_auth):
    with pytest.raises(ValueError):
        admin_auth.set_password("ab")


def test_set_password_rejects_non_ascii(admin_auth):
    assert admin_auth.is_ascii("Passw0rd!-") is True
    assert admin_auth.is_ascii("密码密码") is False
    with pytest.raises(ValueError):
        admin_auth.set_password("中文密码")
    # 仍可正常设置 ASCII 密码
    admin_auth.set_password("ascii-pass")
    assert admin_auth.verify("ascii-pass") is True


def test_password_set_empty_without_env(tmp_path, monkeypatch):
    import src.config as config

    monkeypatch.setattr(config.settings, "admin_token", "", raising=False)
    mod = importlib.import_module("src.api.admin_auth")
    monkeypatch.setattr(mod, "PASSWORD_PATH", tmp_path / "none.json")
    assert mod.password_set() is False
    assert mod.verify("anything") is False
