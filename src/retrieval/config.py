"""Runtime-tunable retrieval parameters.

Environment variables (``src.config.settings``) provide the defaults; this JSON
file overrides them at runtime so the admin UI can tune retrieval without a
restart. Stored at ``qdrant_data/retrieval_config.json``.

Every value is clamped to a sane range so a bad UI/API input cannot make a
query pathological. Chunk parameters take effect on the next import/rebuild.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from src.config import settings

CONFIG_PATH = Path("./qdrant_data/retrieval_config.json")
_lock = threading.RLock()
_cache: dict | None = None

_BOOL_FIELDS = {"hybrid_enabled"}
_INT_FIELDS: dict[str, tuple[int, int]] = {
    "top_k": (1, 100),
    "chunks_per_file": (1, 20),
    "rerank_top_k": (0, 100),
    "rrf_k": (1, 1000),
    "candidate_multiplier": (1, 20),
    "candidate_min": (1, 500),
    "keyword_limit": (0, 20),
    "max_context_chars": (500, 100000),
    "chunk_size": (100, 8000),
    "chunk_overlap": (0, 2000),
}
_FLOAT_FIELDS: dict[str, tuple[float, float]] = {
    "dense_weight": (0.0, 10.0),
    "sparse_weight": (0.0, 10.0),
    "keyword_boost": (0.0, 10.0),
    "min_score": (0.0, 1.0),
}
_FIELDS = [*_INT_FIELDS, *_FLOAT_FIELDS, *_BOOL_FIELDS]


def _defaults() -> dict:
    return {
        "top_k": settings.top_k,
        "chunks_per_file": settings.chunks_per_file,
        "rerank_top_k": settings.rerank_top_k,
        "rrf_k": settings.rrf_k,
        "hybrid_enabled": settings.hybrid_enabled,
        "dense_weight": settings.dense_weight,
        "sparse_weight": settings.sparse_weight,
        "candidate_multiplier": settings.candidate_multiplier,
        "candidate_min": settings.candidate_min,
        "keyword_boost": settings.keyword_boost,
        "keyword_limit": settings.keyword_limit,
        "max_context_chars": settings.max_context_chars,
        "min_score": settings.min_score,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


def _coerce(name: str, value: object) -> int | float | bool | None:
    """Clamp ``value`` for ``name``; return None when it is unusable."""
    if name in _BOOL_FIELDS:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if name in _INT_FIELDS:
        lo, hi = _INT_FIELDS[name]
        try:
            return max(lo, min(hi, int(value)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    lo, hi = _FLOAT_FIELDS[name]
    try:
        return max(lo, min(hi, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize(raw: object) -> dict:
    cfg = _defaults()
    if isinstance(raw, dict):
        for name in _FIELDS:
            if name in raw:
                value = _coerce(name, raw[name])
                if value is not None:
                    cfg[name] = value
    # Overlap must stay below chunk size or the splitter never advances.
    if cfg["chunk_overlap"] >= cfg["chunk_size"]:
        cfg["chunk_overlap"] = max(0, cfg["chunk_size"] // 4)
    return cfg


def load() -> dict:
    global _cache
    with _lock:
        if _cache is not None:
            return dict(_cache)
        raw: object = {}
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        _cache = _normalize(raw)
        return dict(_cache)


def save(patch: dict) -> dict:
    """Merge ``patch`` over the current config, persist and return the result."""
    global _cache
    with _lock:
        current = _cache if _cache is not None else _normalize(_read_raw())
        merged = dict(current)
        for name in _FIELDS:
            if name in patch and patch[name] is not None:
                value = _coerce(name, patch[name])
                if value is not None:
                    merged[name] = value
        if merged["chunk_overlap"] >= merged["chunk_size"]:
            merged["chunk_overlap"] = max(0, merged["chunk_size"] // 4)
        _cache = merged
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        return dict(merged)


def _read_raw() -> object:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def get(name: str, default: int | float | bool | None = None):
    return load().get(name, _defaults().get(name, default))


def chunk_settings() -> tuple[int, int]:
    """(chunk_size, chunk_overlap) currently in effect for new imports."""
    cfg = load()
    return int(cfg["chunk_size"]), int(cfg["chunk_overlap"])
