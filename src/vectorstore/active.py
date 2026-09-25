"""Persisted active-collection selection.

The active collection used to live only in ``settings.qdrant_collection`` and was
mutated at runtime, so a restart/redeploy silently fell back to the env default
(often a collection that does not exist). Persisting the choice under the data
volume keeps the admin's active knowledge base stable across restarts.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from src.config import settings

PATH = Path("./qdrant_data/active_collection.json")
_lock = threading.RLock()


def load() -> str | None:
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = data.get("collection") if isinstance(data, dict) else None
    return name if isinstance(name, str) and name.strip() else None


def save(name: str) -> None:
    if not name:
        return
    with _lock:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(
            json.dumps({"collection": name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def apply() -> None:
    """Adopt the persisted collection into settings when present."""
    name = load()
    if name:
        settings.qdrant_collection = name
