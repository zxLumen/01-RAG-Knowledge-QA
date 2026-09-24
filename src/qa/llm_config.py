"""Persistent LLM / embedding configuration (profiles + active selection).

Stored at ``qdrant_data/llm_config.json``::

    {
      "active_id": "p_xxx",
      "profiles": [
        {"id": "p_xxx", "name": "本地 qwen3", "provider": "ollama",
         "protocol": "ollama", "base_url": "http://localhost:11434",
         "api_key": "", "model": "qwen3:8b", "temperature": 0.0}
      ],
      "embedding": {"provider": "ollama", "protocol": "ollama",
                    "base_url": "http://localhost:11434", "api_key": "",
                    "model": "bge-m3", "dim": 1024}
    }

Environment variables provide the defaults; the JSON file overrides them.
API keys are stored locally in plaintext but never returned by the API (only a
masked hint). ``${ENV_VAR}`` values are resolved from the environment at use
time.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

from src.config import settings
from src.qa import providers

CONFIG_PATH = Path("./qdrant_data/llm_config.json")
LEGACY_MODEL_PATH = Path("./qdrant_data/llm_model.json")

_lock = threading.RLock()
_cache: dict | None = None


def _new_id() -> str:
    return "p_" + uuid.uuid4().hex[:10]


def _auto_name(provider: str, model: str) -> str:
    """Default display name: '<provider name> <model>'."""
    preset = providers.get_provider(provider)
    provider_name = preset["name"] if preset else provider
    return f"{provider_name} {model}".strip()


def resolve_key(value: str | None) -> str:
    """Resolve ``${ENV_VAR}`` references; return the literal value otherwise."""
    if not value:
        return ""
    value = value.strip()
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


def mask_key(value: str | None) -> str:
    """Return a masked hint for a key (never the raw secret)."""
    key = resolve_key(value)
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}****{key[-4:]}"


def _env_default_profile() -> dict:
    provider = settings.llm_provider or "ollama"
    protocol = providers.protocol_for(provider)
    base_url = settings.llm_base_url or providers.default_base_url(provider)
    if provider == "ollama" and not base_url:
        base_url = settings.ollama_base_url
    model = settings.llm_model or settings.ollama_model
    preset = providers.get_provider(provider)
    return {
        "id": _new_id(),
        "name": f"{preset['name'] if preset else provider} {model}".strip(),
        "provider": provider,
        "protocol": protocol,
        "base_url": base_url,
        "api_key": settings.llm_api_key or "",
        "model": model,
        "temperature": settings.llm_temperature,
    }


def _env_default_embedding() -> dict:
    provider = settings.embedding_provider or "ollama"
    protocol = providers.protocol_for(provider)
    base_url = settings.embedding_base_url or providers.default_base_url(provider)
    if provider == "ollama" and not base_url:
        base_url = settings.ollama_base_url
    model = settings.embedding_model or settings.dense_embedding_model
    return {
        "id": _new_id(),
        "name": _auto_name(provider, model),
        "provider": provider,
        "protocol": protocol,
        "base_url": base_url,
        "api_key": settings.embedding_api_key or "",
        "model": model,
        "dim": settings.embedding_dim,
    }


def _migrate_legacy() -> dict | None:
    """Build a default profile from the legacy ``llm_model.json`` if present."""
    try:
        data = json.loads(LEGACY_MODEL_PATH.read_text(encoding="utf-8"))
        model = data.get("model")
    except (OSError, ValueError):
        return None
    if not isinstance(model, str) or not model:
        return None
    profile = _env_default_profile()
    profile["provider"] = "ollama"
    profile["protocol"] = providers.PROTOCOL_OLLAMA
    profile["base_url"] = settings.ollama_base_url
    profile["model"] = model
    profile["name"] = _auto_name("ollama", model)
    return profile


def _default_config() -> dict:
    profile = _migrate_legacy() or _env_default_profile()
    embedding = _env_default_embedding()
    return {
        "active_id": profile["id"],
        "profiles": [profile],
        "embedding_active_id": embedding["id"],
        "embedding_profiles": [embedding],
    }


def _normalize(raw: dict) -> dict:
    cfg = _default_config()
    if isinstance(raw, dict):
        if isinstance(raw.get("profiles"), list):
            profiles = [p for p in raw["profiles"] if isinstance(p, dict) and p.get("id")]
            if profiles:
                cfg["profiles"] = profiles
        if isinstance(raw.get("active_id"), str) and raw["active_id"]:
            cfg["active_id"] = raw["active_id"]
        emb_profiles = raw.get("embedding_profiles")
        if isinstance(emb_profiles, list):
            valid = [p for p in emb_profiles if isinstance(p, dict) and p.get("id")]
            if valid:
                cfg["embedding_profiles"] = valid
        elif isinstance(raw.get("embedding"), dict):
            # Migrate legacy single embedding object into a profile.
            legacy = dict(cfg["embedding_profiles"][0])
            had_name = bool(raw["embedding"].get("name"))
            legacy.update(raw["embedding"])
            legacy.setdefault("id", _new_id())
            if not had_name:
                legacy["name"] = ""
            cfg["embedding_profiles"] = [legacy]
        if isinstance(raw.get("embedding_active_id"), str) and raw["embedding_active_id"]:
            cfg["embedding_active_id"] = raw["embedding_active_id"]

    ids = {p["id"] for p in cfg["profiles"]}
    if cfg["active_id"] not in ids:
        cfg["active_id"] = cfg["profiles"][0]["id"]
    for p in cfg["profiles"]:
        p.setdefault("protocol", providers.protocol_for(p.get("provider")))
        p.setdefault("base_url", providers.default_base_url(p.get("provider")))
        p.setdefault("api_key", "")
        p.setdefault("temperature", 0.0)

    emb_ids = {p["id"] for p in cfg["embedding_profiles"]}
    if cfg["embedding_active_id"] not in emb_ids:
        cfg["embedding_active_id"] = cfg["embedding_profiles"][0]["id"]
    for e in cfg["embedding_profiles"]:
        e.setdefault("protocol", providers.protocol_for(e.get("provider")))
        e.setdefault("base_url", providers.default_base_url(e.get("provider")))
        e.setdefault("api_key", "")
        e.setdefault("dim", settings.embedding_dim)
        if not e.get("name"):
            e["name"] = _auto_name(e.get("provider", ""), e.get("model", ""))
    return cfg


def load() -> dict:
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        raw: dict = {}
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        _cache = _normalize(raw)
        return _cache


def _persist(cfg: dict) -> None:
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def save(cfg: dict) -> None:
    global _cache
    with _lock:
        _cache = cfg
        _persist(cfg)


# --- profiles -------------------------------------------------------------


def list_profiles() -> list[dict]:
    return load()["profiles"]


def get_profile(profile_id: str | None) -> dict | None:
    if not profile_id:
        return None
    for p in load()["profiles"]:
        if p["id"] == profile_id:
            return p
    return None


def get_active() -> dict:
    cfg = load()
    return get_profile(cfg["active_id"]) or cfg["profiles"][0]


def upsert_profile(data: dict) -> dict:
    cfg = load()
    profile_id = (data.get("id") or "").strip()
    provider = (data.get("provider") or "custom").strip()
    protocol = providers.protocol_for(provider)
    base_url = (data.get("base_url") or "").strip() or providers.default_base_url(provider)
    model = (data.get("model") or "").strip()
    temperature = float(data.get("temperature") or 0.0)
    name = (data.get("name") or "").strip()

    existing = get_profile(profile_id) if profile_id else None
    api_key = data.get("api_key")
    if api_key is None or (isinstance(api_key, str) and not api_key.strip()):
        # Empty means "keep existing key" when editing.
        resolved_key = existing.get("api_key", "") if existing else ""
    else:
        resolved_key = api_key.strip()

    if existing:
        existing.update(
            {
                "name": name or existing.get("name") or _auto_name(provider, model),
                "provider": provider,
                "protocol": protocol,
                "base_url": base_url,
                "api_key": resolved_key,
                "model": model,
                "temperature": temperature,
            }
        )
        profile = existing
    else:
        profile = {
            "id": profile_id or _new_id(),
            "name": name or _auto_name(provider, model),
            "provider": provider,
            "protocol": protocol,
            "base_url": base_url,
            "api_key": resolved_key,
            "model": model,
            "temperature": temperature,
        }
        cfg["profiles"].append(profile)
    save(cfg)
    return profile


def delete_profile(profile_id: str) -> bool:
    cfg = load()
    if len(cfg["profiles"]) <= 1:
        return False
    remaining = [p for p in cfg["profiles"] if p["id"] != profile_id]
    if len(remaining) == len(cfg["profiles"]):
        return False
    cfg["profiles"] = remaining
    if cfg["active_id"] == profile_id:
        cfg["active_id"] = remaining[0]["id"]
    save(cfg)
    return True


def set_active(profile_id: str) -> bool:
    cfg = load()
    if not get_profile(profile_id):
        return False
    cfg["active_id"] = profile_id
    save(cfg)
    return True


# --- embedding profiles ---------------------------------------------------


def list_embedding_profiles() -> list[dict]:
    return load()["embedding_profiles"]


def get_embedding_profile(profile_id: str | None) -> dict | None:
    if not profile_id:
        return None
    for p in load()["embedding_profiles"]:
        if p["id"] == profile_id:
            return p
    return None


def get_embedding_active() -> dict:
    cfg = load()
    return get_embedding_profile(cfg["embedding_active_id"]) or cfg["embedding_profiles"][0]


def upsert_embedding_profile(data: dict) -> dict:
    cfg = load()
    profile_id = (data.get("id") or "").strip()
    provider = (data.get("provider") or "ollama").strip()
    protocol = providers.protocol_for(provider)
    base_url = (data.get("base_url") or "").strip() or providers.default_base_url(provider)
    model = (data.get("model") or "").strip()
    name = (data.get("name") or "").strip()

    existing = get_embedding_profile(profile_id) if profile_id else None
    api_key = data.get("api_key")
    if api_key is None or (isinstance(api_key, str) and not api_key.strip()):
        resolved_key = existing.get("api_key", "") if existing else ""
    else:
        resolved_key = api_key.strip()

    dim = data.get("dim") or (existing.get("dim") if existing else None)
    if not dim:
        dim = providers.suggest_dim(model) or settings.embedding_dim

    if existing:
        existing.update(
            {
                "name": name or existing.get("name") or _auto_name(provider, model),
                "provider": provider,
                "protocol": protocol,
                "base_url": base_url,
                "api_key": resolved_key,
                "model": model,
                "dim": int(dim),
            }
        )
        profile = existing
    else:
        profile = {
            "id": profile_id or _new_id(),
            "name": name or _auto_name(provider, model),
            "provider": provider,
            "protocol": protocol,
            "base_url": base_url,
            "api_key": resolved_key,
            "model": model,
            "dim": int(dim),
        }
        cfg["embedding_profiles"].append(profile)
    save(cfg)
    return profile


def delete_embedding_profile(profile_id: str) -> bool:
    cfg = load()
    if len(cfg["embedding_profiles"]) <= 1:
        return False
    remaining = [p for p in cfg["embedding_profiles"] if p["id"] != profile_id]
    if len(remaining) == len(cfg["embedding_profiles"]):
        return False
    cfg["embedding_profiles"] = remaining
    if cfg["embedding_active_id"] == profile_id:
        cfg["embedding_active_id"] = remaining[0]["id"]
    save(cfg)
    return True


def set_embedding_active(profile_id: str) -> bool:
    cfg = load()
    if not get_embedding_profile(profile_id):
        return False
    cfg["embedding_active_id"] = profile_id
    save(cfg)
    return True


def get_embedding() -> dict:
    return get_embedding_active()


def embedding_dim() -> int:
    try:
        return int(get_embedding().get("dim") or settings.embedding_dim)
    except (TypeError, ValueError):
        return settings.embedding_dim


# --- public (masked) serialization ---------------------------------------


def public_profile(p: dict) -> dict:
    out = dict(p)
    out["api_key"] = ""
    out["has_key"] = bool(resolve_key(p.get("api_key")))
    out["key_hint"] = mask_key(p.get("api_key"))
    return out


def public_config() -> dict:
    cfg = load()
    return {
        "active_id": cfg["active_id"],
        "profiles": [public_profile(p) for p in cfg["profiles"]],
        "embedding_active_id": cfg["embedding_active_id"],
        "embedding_profiles": [public_profile(p) for p in cfg["embedding_profiles"]],
    }


# --- backward compatibility ----------------------------------------------


def get_current_model() -> str:
    return get_active().get("model", "")


def set_current_model(model: str) -> str:
    active = get_active()
    upsert_profile({"id": active["id"], "model": model})
    return model
