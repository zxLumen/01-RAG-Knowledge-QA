from __future__ import annotations

import pytest

from src.qa import llm_config, providers


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_config, "CONFIG_PATH", tmp_path / "llm_config.json")
    monkeypatch.setattr(llm_config, "LEGACY_MODEL_PATH", tmp_path / "llm_model.json")
    monkeypatch.setattr(llm_config, "_cache", None)
    yield llm_config
    monkeypatch.setattr(llm_config, "_cache", None)


def test_default_has_one_active_profile(cfg):
    data = cfg.public_config()
    assert len(data["profiles"]) == 1
    assert data["active_id"] == data["profiles"][0]["id"]
    assert len(data["embedding_profiles"]) == 1
    assert data["embedding_active_id"] == data["embedding_profiles"][0]["id"]
    assert data["embedding_profiles"][0]["model"]


def test_upsert_and_switch_active(cfg):
    p = cfg.upsert_profile(
        {
            "name": "DeepSeek",
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": "sk-abcdef123456",
        }
    )
    assert p["protocol"] == "openai"
    assert p["base_url"] == providers.default_base_url("deepseek")
    assert cfg.set_active(p["id"]) is True
    assert cfg.get_active()["id"] == p["id"]


def test_api_key_masked_and_never_returned(cfg):
    cfg.upsert_profile({"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-abcdef123456"})
    pub = cfg.public_config()
    target = next(p for p in pub["profiles"] if p["provider"] == "openai")
    assert target["api_key"] == ""
    assert target["has_key"] is True
    assert target["key_hint"] == "sk-a****3456"
    assert "sk-abcdef123456" not in str(pub)


def test_empty_key_keeps_existing(cfg):
    p = cfg.upsert_profile(
        {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-secret-key"}
    )
    cfg.upsert_profile({"id": p["id"], "model": "gpt-4o", "api_key": ""})
    assert cfg.get_profile(p["id"])["api_key"] == "sk-secret-key"
    assert cfg.get_profile(p["id"])["model"] == "gpt-4o"


def test_delete_keeps_at_least_one(cfg):
    only = cfg.get_active()["id"]
    assert cfg.delete_profile(only) is False
    p = cfg.upsert_profile({"provider": "deepseek", "model": "deepseek-chat"})
    assert cfg.delete_profile(p["id"]) is True


def test_resolve_key_env(monkeypatch):
    monkeypatch.setenv("MY_LLM_KEY", "sk-from-env")
    assert cfg_resolve("${MY_LLM_KEY}") == "sk-from-env"
    assert cfg_resolve("sk-literal") == "sk-literal"
    assert cfg_resolve("") == ""


def cfg_resolve(value: str) -> str:
    return llm_config.resolve_key(value)


def test_embedding_update_keeps_key(cfg):
    p = cfg.upsert_embedding_profile(
        {"provider": "openai", "model": "text-embedding-3-small", "api_key": "sk-emb", "dim": 1536}
    )
    cfg.set_embedding_active(p["id"])
    cfg.upsert_embedding_profile(
        {"id": p["id"], "provider": "openai", "model": "text-embedding-3-large", "dim": 3072}
    )
    emb = cfg.get_embedding()
    assert emb["model"] == "text-embedding-3-large"
    assert emb["dim"] == 3072
    assert emb["api_key"] == "sk-emb"
    assert cfg.embedding_dim() == 3072


def test_embedding_profile_crud_and_switch(cfg):
    only = cfg.get_embedding_active()["id"]
    assert cfg.delete_embedding_profile(only) is False
    p = cfg.upsert_embedding_profile(
        {"name": "硅基 bge", "provider": "siliconflow", "model": "BAAI/bge-m3"}
    )
    assert p["dim"] == 1024  # auto from known dims
    assert cfg.set_embedding_active(p["id"]) is True
    assert cfg.get_embedding_active()["id"] == p["id"]
    assert cfg.delete_embedding_profile(p["id"]) is True


def test_legacy_embedding_object_migrates(cfg, tmp_path, monkeypatch):
    import json

    cfg.CONFIG_PATH.write_text(
        json.dumps(
            {
                "active_id": "p_x",
                "profiles": [{"id": "p_x", "provider": "ollama", "model": "qwen3:8b"}],
                "embedding": {"provider": "openai", "model": "text-embedding-3-large", "dim": 3072},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_config, "_cache", None)
    emb = cfg.get_embedding()
    assert emb["provider"] == "openai"
    assert emb["model"] == "text-embedding-3-large"
    assert emb["dim"] == 3072
    assert len(cfg.list_embedding_profiles()) == 1


def test_provider_helpers():
    assert providers.protocol_for("ollama") == "ollama"
    assert providers.protocol_for("deepseek") == "openai"
    assert providers.protocol_for("unknown") == "openai"
    assert providers.suggest_dim("BAAI/bge-m3") == 1024
    assert providers.suggest_dim("text-embedding-3-small") == 1536
    assert providers.suggest_dim("nope") is None
