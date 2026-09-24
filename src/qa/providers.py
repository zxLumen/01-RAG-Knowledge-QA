"""Built-in LLM / embedding provider presets.

Two wire protocols are supported:

- ``ollama``  native Ollama API (``/api/chat``, ``/api/embed``)
- ``openai``  OpenAI-compatible API (``/chat/completions``, ``/embeddings``)

Most vendors expose an OpenAI-compatible endpoint; ``base_url`` should point at
the ``/v1`` root (no trailing slash). Model lists are always fetched live from
the provider (a valid API key is required); no preset fallback is used.
"""

from __future__ import annotations

PROTOCOL_OLLAMA = "ollama"
PROTOCOL_OPENAI = "openai"
PROTOCOLS = (PROTOCOL_OLLAMA, PROTOCOL_OPENAI)

# id -> preset. ``embeddings`` marks vendors that expose an embeddings API.
PROVIDERS: list[dict] = [
    {
        "id": "ollama",
        "name": "本地 Ollama",
        "protocol": PROTOCOL_OLLAMA,
        "base_url": "http://localhost:11434",
        "embeddings": True,
    },
    {
        "id": "openai",
        "name": "OpenAI",
        "protocol": PROTOCOL_OPENAI,
        "base_url": "https://api.openai.com/v1",
        "embeddings": True,
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "protocol": PROTOCOL_OPENAI,
        "base_url": "https://api.deepseek.com/v1",
        "embeddings": False,
    },
    {
        "id": "zhipuai",
        "name": "智谱 GLM",
        "protocol": PROTOCOL_OPENAI,
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "embeddings": True,
    },
    {
        "id": "dashscope",
        "name": "通义千问（阿里云百炼）",
        "protocol": PROTOCOL_OPENAI,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embeddings": True,
    },
    {
        "id": "moonshot",
        "name": "Moonshot / Kimi",
        "protocol": PROTOCOL_OPENAI,
        "base_url": "https://api.moonshot.cn/v1",
        "embeddings": False,
    },
    {
        "id": "siliconflow",
        "name": "硅基流动 SiliconFlow",
        "protocol": PROTOCOL_OPENAI,
        "base_url": "https://api.siliconflow.cn/v1",
        "embeddings": True,
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "protocol": PROTOCOL_OPENAI,
        "base_url": "https://openrouter.ai/api/v1",
        "embeddings": False,
    },
    {
        "id": "opencode",
        "name": "OpenCode Zen",
        "protocol": PROTOCOL_OPENAI,
        "base_url": "https://opencode.ai/zen/v1",
        "env": "OPENCODE_API_KEY",
        "embeddings": False,
    },
    {
        "id": "opencode-go",
        "name": "OpenCode Go",
        "protocol": PROTOCOL_OPENAI,
        "base_url": "https://opencode.ai/zen/go/v1",
        "env": "OPENCODE_API_KEY",
        "embeddings": False,
    },
    {
        "id": "custom",
        "name": "自定义（OpenAI 兼容）",
        "protocol": PROTOCOL_OPENAI,
        "base_url": "",
        "embeddings": True,
    },
]

_BY_ID = {p["id"]: p for p in PROVIDERS}

# Known output dimensions for common embedding models (used to pre-fill `dim`).
KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "bge-m3": 1024,
    "baai/bge-m3": 1024,
    "pro/baai/bge-m3": 1024,
    "nomic-embed-text": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "embedding-3": 2048,
    "embedding-2": 1024,
    "text-embedding-v3": 1024,
    "text-embedding-v4": 1024,
    "gemini-embedding-001": 3072,
}


def list_providers() -> list[dict]:
    return PROVIDERS


def list_embedding_providers() -> list[dict]:
    return [p for p in PROVIDERS if p.get("embeddings")]


def get_provider(provider_id: str | None) -> dict | None:
    if not provider_id:
        return None
    return _BY_ID.get(provider_id)


def supports_embeddings(provider_id: str | None) -> bool:
    preset = get_provider(provider_id)
    return bool(preset and preset.get("embeddings"))


def protocol_for(provider_id: str | None, default: str = PROTOCOL_OPENAI) -> str:
    preset = get_provider(provider_id)
    if preset:
        return preset["protocol"]
    return default


def default_base_url(provider_id: str | None) -> str:
    preset = get_provider(provider_id)
    return preset["base_url"] if preset else ""


def suggest_dim(model: str | None) -> int | None:
    if not model:
        return None
    return KNOWN_EMBEDDING_DIMS.get(model.strip().lower())


_EMBEDDING_HINTS = ("embed", "bge", "gte", "e5", "nomic", "text-embedding")


def is_embedding_model(model: str | None) -> bool:
    """Heuristic: does this model name look like an embedding model?"""
    if not model:
        return False
    name = model.lower()
    return any(hint in name for hint in _EMBEDDING_HINTS)


def filter_by_purpose(models: list[str], purpose: str = "chat") -> list[str]:
    """Split a provider model list into chat vs embedding models."""
    want_embedding = purpose == "embedding"
    return [m for m in models if is_embedding_model(m) == want_embedding]
