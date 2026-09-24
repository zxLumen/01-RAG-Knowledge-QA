from __future__ import annotations

from src.qa import providers


def test_opencode_presets_present():
    ids = {p["id"] for p in providers.list_providers()}
    assert "opencode" in ids
    assert "opencode-go" in ids
    assert providers.get_provider("opencode")["base_url"] == "https://opencode.ai/zen/v1"
    assert providers.get_provider("opencode-go")["base_url"] == "https://opencode.ai/zen/go/v1"
    assert providers.get_provider("opencode")["protocol"] == "openai"


def test_is_embedding_model():
    assert providers.is_embedding_model("BAAI/bge-m3")
    assert providers.is_embedding_model("text-embedding-3-small")
    assert not providers.is_embedding_model("deepseek-chat")
    assert not providers.is_embedding_model("gpt-4o")


def test_filter_by_purpose():
    models = ["qwen3:8b", "bge-m3", "text-embedding-3-small", "deepseek-chat"]
    emb = providers.filter_by_purpose(models, "embedding")
    chat = providers.filter_by_purpose(models, "chat")
    assert "bge-m3" in emb and "text-embedding-3-small" in emb
    assert "qwen3:8b" in chat and "deepseek-chat" in chat
    assert "bge-m3" not in chat


def test_presets_have_no_model_lists():
    for p in providers.list_providers():
        assert "models" not in p


def test_embeddings_flag_and_filter():
    assert providers.supports_embeddings("ollama")
    assert providers.supports_embeddings("openai")
    assert providers.supports_embeddings("siliconflow")
    assert not providers.supports_embeddings("deepseek")
    assert not providers.supports_embeddings("opencode")
    emb_ids = {p["id"] for p in providers.list_embedding_providers()}
    assert "deepseek" not in emb_ids
    assert "openai" in emb_ids
    assert "custom" in emb_ids

