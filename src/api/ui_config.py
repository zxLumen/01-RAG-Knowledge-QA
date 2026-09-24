"""Global UI configuration (theme allowlist and ordering).

Stored at ``qdrant_data/ui_config.json``. Read publicly by the web UI, written
only by an authenticated admin.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

CONFIG_PATH = Path("./qdrant_data/ui_config.json")
_lock = threading.RLock()

ALL_THEMES = ["default", "purple", "emerald", "amber", "light"]


def _default() -> dict:
    return {"themes": list(ALL_THEMES), "default_theme": "default"}


def load() -> dict:
    cfg = _default()
    with _lock:
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cfg
    themes = raw.get("themes")
    if isinstance(themes, list):
        valid = [t for t in themes if isinstance(t, str) and t in ALL_THEMES]
        if valid:
            cfg["themes"] = list(dict.fromkeys(valid))
    default_theme = raw.get("default_theme")
    if isinstance(default_theme, str) and default_theme in cfg["themes"]:
        cfg["default_theme"] = default_theme
    else:
        cfg["default_theme"] = cfg["themes"][0]
    return cfg


def save(cfg: dict) -> dict:
    clean = _default()
    themes = cfg.get("themes")
    if isinstance(themes, list):
        valid = [t for t in themes if isinstance(t, str) and t in ALL_THEMES]
        if valid:
            clean["themes"] = list(dict.fromkeys(valid))
    default_theme = cfg.get("default_theme")
    if isinstance(default_theme, str) and default_theme in clean["themes"]:
        clean["default_theme"] = default_theme
    else:
        clean["default_theme"] = clean["themes"][0]
    with _lock:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return clean
