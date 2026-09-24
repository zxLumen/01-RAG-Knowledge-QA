from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_ALIAS_FILE = Path("data/collection_aliases.json")
_ORDER_FILE = Path("data/collection_order.json")
_STORAGE_RE = re.compile(r"^[A-Za-z0-9_-]{1,63}$")


def _load_aliases() -> dict[str, str]:
    try:
        return json.loads(_ALIAS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_aliases(aliases: dict[str, str]) -> None:
    _ALIAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ALIAS_FILE.write_text(
        json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def storage_name(display: str) -> str:
    """Qdrant-legal collection name for a (possibly non-ASCII) display name."""
    d = (display or "").strip()
    if _STORAGE_RE.match(d):
        return d
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", d)
    slug = re.sub(r"_+", "_", slug).strip("_")
    digest = hashlib.md5(d.encode("utf-8")).hexdigest()[:6]
    base = slug[:42] if slug else "col"
    return f"{base}_{digest}"[:63]


def register_alias(display: str, storage: str) -> None:
    """Persist display->storage mapping when they differ (ASCII names are identity)."""
    d = (display or "").strip()
    if d == storage:
        return
    aliases = _load_aliases()
    if aliases.get(d) == storage:
        return
    aliases[d] = storage
    _save_aliases(aliases)


def display_name(storage: str) -> str:
    """Human-facing collection name; falls back to the storage name."""
    for display, s in _load_aliases().items():
        if s == storage:
            return display
    return storage


def storage_for_display(display: str) -> str:
    """Map a display collection name back to its Qdrant storage name."""
    d = (display or "").strip()
    for disp, s in _load_aliases().items():
        if disp == d:
            return s
    return d


def rename_display(current: str, new_display: str) -> None:
    """Rename the human-facing name of a collection; Qdrant name is unchanged.

    Raises ValueError if the new name is blank. Pass the current storage name to
    restore the identity display (i.e. drop the alias).
    """
    storage = storage_for_display(current)
    new_display = (new_display or "").strip()
    if not new_display:
        raise ValueError("新的集合名称不能为空")
    aliases = _load_aliases()
    aliases = {d: s for d, s in aliases.items() if s != storage}
    if new_display != storage:
        aliases[new_display] = storage
    _save_aliases(aliases)


def delete_alias(storage: str) -> None:
    """Drop every alias pointing at a collection about to be deleted."""
    aliases = _load_aliases()
    kept = {d: s for d, s in aliases.items() if s != storage}
    if len(kept) != len(aliases):
        _save_aliases(kept)


def _load_order() -> list[str]:
    try:
        data = json.loads(_ORDER_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [s for s in data if isinstance(s, str)] if isinstance(data, list) else []


def _save_order(order: list[str]) -> None:
    _ORDER_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ORDER_FILE.write_text(
        json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def record_creation(storage: str) -> None:
    """Remember a collection's creation order (append only once)."""
    order = _load_order()
    if storage in order:
        return
    order.append(storage)
    _save_order(order)


def drop_creation(storage: str) -> None:
    """Forget a deleted collection so it does not linger in the ordering."""
    order = _load_order()
    kept = [s for s in order if s != storage]
    if len(kept) != len(order):
        _save_order(kept)


def order_index() -> dict[str, int]:
    """Map storage name -> position in creation order."""
    return {s: i for i, s in enumerate(_load_order())}

