"""Server-side persistence for chat conversations.

Chat history is only kept in the browser by default (localStorage), which means
deleted conversations are unrecoverable. This store mirrors every conversation
server-side so administrators can inspect, soft-delete and restore them and so a
recovering browser tab can pull its history back after a restore.

"Sessions" here are *chat* conversations (as opposed to the import ledger in
``src.imports.store``). Each row is keyed by (owner, id) where ``id`` is the
frontend-generated chat session id and ``owner`` is either ``"admin"`` or the
visitor's dedicated collection name (e.g. ``visitor_<id>``).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from src.config import settings

_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL,
    owner TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    collection TEXT NOT NULL DEFAULT '',
    messages TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    UNIQUE(owner, id)
);

CREATE INDEX IF NOT EXISTS idx_chat_owner ON chat_sessions(owner);
CREATE INDEX IF NOT EXISTS idx_chat_updated ON chat_sessions(updated_at DESC);
"""

ADMIN_OWNER = "admin"

# Arbitrary safety caps for a single payload; large histories are synced whole.
MAX_MESSAGES = 500
MAX_MESSAGE_BYTES = 1_000_000


def _connect() -> sqlite3.Connection:
    path = Path(settings.chat_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    with _LOCK:
        conn = _connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db() -> None:
    with _connection() as conn:
        conn.executescript(_SCHEMA)


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    try:
        data["messages"] = json.loads(data["messages"])
    except (ValueError, TypeError):
        data["messages"] = []
    data["deleted"] = bool(data["deleted"])
    return data


def upsert(
    owner: str,
    session_id: str,
    *,
    title: str = "",
    collection: str = "",
    messages: list | None = None,
    deleted: bool = False,
) -> dict:
    """Create or update a chat session.

    A normal sync (``deleted=False``) reactivates a soft-deleted row so ongoing
    conversations are never hidden. Marking a row deleted keeps its contents for
    later restore (admin).
    """
    init_db()
    now = _ts()
    payload = json.dumps(messages or [], ensure_ascii=False)
    title = (title or "")[:500]
    collection = (collection or "")[:200]
    with _connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (
                id, owner, title, collection, messages, created_at,
                updated_at, deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner, id) DO UPDATE SET
                title = CASE WHEN excluded.title != ''
                    THEN excluded.title ELSE chat_sessions.title END,
                collection = CASE WHEN excluded.collection != ''
                    THEN excluded.collection ELSE chat_sessions.collection END,
                messages = CASE WHEN ? = 0
                    THEN excluded.messages ELSE chat_sessions.messages END,
                updated_at = excluded.updated_at,
                deleted = excluded.deleted
            """,
            (
                session_id,
                owner,
                title,
                collection,
                payload,
                now,
                now,
                int(deleted),
                int(deleted),
            ),
        )
        row = conn.execute(
            "SELECT * FROM chat_sessions WHERE owner = ? AND id = ?",
            (owner, session_id),
        ).fetchone()
        return _row_to_dict(row)


def list_for_owner(owner: str, *, include_deleted: bool = False, limit: int = 100) -> list[dict]:
    init_db()
    where = "WHERE owner = ?" + ("" if include_deleted else " AND deleted = 0")
    with _connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM chat_sessions {where} ORDER BY updated_at DESC, rowid DESC LIMIT ?",
            (owner, limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def admin_list(limit: int = 500) -> list[dict]:
    """All chat sessions across every owner, newest first (admin panel)."""
    init_db()
    with _connection() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_sessions ORDER BY updated_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_by_rowid(rowid: int) -> dict | None:
    init_db()
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM chat_sessions WHERE rowid = ?", (rowid,)
        ).fetchone()
        return _row_to_dict(row) if row else None


def set_deleted(rowid: int, deleted: bool) -> bool:
    init_db()
    with _connection() as conn:
        cur = conn.execute(
            "UPDATE chat_sessions SET deleted = ?, updated_at = ? WHERE rowid = ?",
            (int(deleted), _ts(), rowid),
        )
        return cur.rowcount > 0


def hard_delete(rowid: int) -> bool:
    init_db()
    with _connection() as conn:
        cur = conn.execute("DELETE FROM chat_sessions WHERE rowid = ?", (rowid,))
        return cur.rowcount > 0


def purge_owner(owner: str) -> int:
    """Drop every row owned by ``owner`` (used when a visitor is evicted)."""
    init_db()
    with _connection() as conn:
        cur = conn.execute("DELETE FROM chat_sessions WHERE owner = ?", (owner,))
        return cur.rowcount
