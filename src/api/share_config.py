"""Admin-configured resources shared with all visitors (read-only).

Stored at ``qdrant_data/share_config.json`` (persisted with the data volume):

- ``collections``: display names of knowledge bases any visitor may select and
  query (read-only).
- ``paths``: ``data/``-relative files or directories made visible to every
  visitor in the import page. Visitors can read and import them into their own
  knowledge base, but never modify them.

The read-only stance is intentional: shared resources are curated by the admin.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

CONFIG_PATH = Path("./qdrant_data/share_config.json")
_lock = threading.RLock()


def _default() -> dict:
    return {"collections": [], "paths": []}


def _clean_list(values: object) -> list[str]:
    out: list[str] = []
    if isinstance(values, list):
        for v in values:
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
    return list(dict.fromkeys(out))


def load() -> dict:
    cfg = _default()
    with _lock:
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cfg
    if isinstance(raw, dict):
        cfg["collections"] = _clean_list(raw.get("collections"))
        cfg["paths"] = _clean_list(raw.get("paths"))
    return cfg


def save(cfg: dict) -> dict:
    clean = {
        "collections": _clean_list(cfg.get("collections")),
        "paths": _clean_list(cfg.get("paths")),
    }
    with _lock:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return clean


def shared_collections() -> list[str]:
    return load()["collections"]


def shared_paths() -> list[str]:
    return load()["paths"]
