"""Per-request active collection (contextvar).

Visitors are isolated into their own Qdrant collection. Rather than mutating the
global settings (unsafe under concurrency), requests set a contextvar that store
and ingest code read as the effective collection.
"""

from __future__ import annotations

import contextvars

from src.config import settings

_active_collection: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_collection", default=None
)


def set_collection(name: str | None) -> contextvars.Token:
    return _active_collection.set(name)


def reset_collection(token: contextvars.Token) -> None:
    _active_collection.reset(token)


def current_collection() -> str:
    return _active_collection.get() or settings.qdrant_collection
