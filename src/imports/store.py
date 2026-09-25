from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from src.config import settings

_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    path TEXT NOT NULL,
    recreate INTEGER NOT NULL DEFAULT 0,
    documents INTEGER NOT NULL DEFAULT 0,
    chunks INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error TEXT,
    duration_ms INTEGER,
    embedding_model TEXT,
    chunk_size INTEGER,
    chunk_overlap INTEGER,
    collection TEXT NOT NULL DEFAULT 'knowledge_base'
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    status TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    file_size INTEGER NOT NULL DEFAULT 0,
    file_md5 TEXT
);

CREATE INDEX IF NOT EXISTS idx_files_filename ON files(filename);
CREATE INDEX IF NOT EXISTS idx_files_session ON files(session_id);
"""


def _connect() -> sqlite3.Connection:
    path = Path(settings.import_db_path)
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
        _migrate_sessions_collection(conn)


def _migrate_sessions_collection(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    if "collection" not in cols:
        conn.execute(
            "ALTER TABLE sessions ADD COLUMN collection TEXT NOT NULL DEFAULT 'knowledge_base'"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_collection ON sessions(collection)"
    )


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def add_session(
    path: str,
    *,
    recreate: bool,
    documents: int,
    chunks: int,
    status: str,
    error: str | None = None,
    duration_ms: int | None = None,
    embedding_model: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    collection: str | None = None,
) -> int:
    init_db()
    collection = collection or settings.qdrant_collection
    with _connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO sessions (
                ts, path, recreate, documents, chunks, status, error,
                duration_ms, embedding_model, chunk_size, chunk_overlap, collection
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _ts(),
                path,
                int(recreate),
                documents,
                chunks,
                status,
                error,
                duration_ms,
                embedding_model,
                chunk_size,
                chunk_overlap,
                collection,
            ),
        )
        return int(cur.lastrowid)


def add_files(session_id: int, files: list[dict]) -> None:
    init_db()
    with _connection() as conn:
        conn.executemany(
            """
            INSERT INTO files (
                session_id, filename, rel_path, status, chunk_count, file_size, file_md5
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    f["filename"],
                    f["rel_path"],
                    f["status"],
                    f["chunk_count"],
                    f["file_size"],
                    f["file_md5"],
                )
                for f in files
            ],
        )


def list_sessions(
    q: str | None = None,
    limit: int = 100,
    offset: int = 0,
    collection: str | None = None,
) -> list[dict]:
    init_db()
    collection = collection or settings.qdrant_collection
    where = "WHERE collection = ?"
    params: list = [collection]
    if q:
        where += " AND path LIKE ?"
        params.append(f"%{q}%")
    with _connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM sessions
            {where}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()
        return [dict(r) for r in rows]


def delete_collection_sessions(collection: str) -> int:
    """Delete every import record for a collection; returns rows removed."""
    init_db()
    with _connection() as conn:
        conn.execute(
            "DELETE FROM files WHERE session_id IN (SELECT id FROM sessions WHERE collection = ?)",
            (collection,),
        )
        cur = conn.execute("DELETE FROM sessions WHERE collection = ?", (collection,))
        conn.commit()
        return cur.rowcount


def get_session(session_id: int) -> dict | None:
    init_db()
    with _connection() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


def list_files(session_id: int) -> list[dict]:
    init_db()
    with _connection() as conn:
        rows = conn.execute(
            "SELECT * FROM files WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def latest_md5_by_filename() -> dict[str, dict]:
    """Return {filename: {md5, status, session_id}} from the newest session of each file."""
    init_db()
    with _connection() as conn:
        rows = conn.execute(
            """
            SELECT f.filename, f.file_md5, f.status, f.session_id
            FROM files f
            JOIN (
                SELECT filename, MAX(id) AS max_id
                FROM files
                GROUP BY filename
            ) latest ON latest.max_id = f.id
            """
        ).fetchall()
        return {r["filename"]: dict(r) for r in rows}


def latest_md5_by_rel_path(collection: str | None = None) -> dict[str, dict]:
    """Return {rel_path: {md5, status, filename, chunk_count}} from the newest
    row per rel_path within the given (or active) collection."""
    init_db()
    collection = collection or settings.qdrant_collection
    with _connection() as conn:
        rows = conn.execute(
            """
            SELECT f.rel_path, f.file_md5, f.status, f.filename, f.chunk_count
            FROM files f
            JOIN sessions s ON s.id = f.session_id
            JOIN (
                SELECT f2.rel_path, MAX(f2.id) AS max_id
                FROM files f2
                JOIN sessions s2 ON s2.id = f2.session_id
                WHERE s2.collection = ?
                GROUP BY f2.rel_path
            ) latest ON latest.max_id = f.id
            WHERE s.collection = ?
            """,
            (collection, collection),
        ).fetchall()
        return {r["rel_path"]: dict(r) for r in rows}


def prune_files(collection: str | None, keep_sources: set[str]) -> int:
    """Drop file ledger rows of a collection whose rel_path is not being
    (re)indexed. Keeps sessions for history. Mirrors recreate wiping the
    vector collection so stale md5 guesses don't skip later imports."""
    init_db()
    collection = collection or settings.qdrant_collection
    with _connection() as conn:
        if not keep_sources:
            cur = conn.execute(
                """
                DELETE FROM files WHERE session_id IN (
                    SELECT id FROM sessions WHERE collection = ?
                )
                """,
                (collection,),
            )
        else:
            placeholders = ", ".join("?" for _ in keep_sources)
            cur = conn.execute(
                f"""
                DELETE FROM files WHERE session_id IN (
                    SELECT id FROM sessions WHERE collection = ?
                ) AND rel_path NOT IN ({placeholders})
                """,
                [collection, *sorted(keep_sources)],
            )
        return int(cur.rowcount)


def record_deletion(collection: str, rel_paths: list[str]) -> int:
    """Record manually deleted files as tombstones in the import ledger.

    A synthetic session is created whose file rows carry ``file_md5 = NULL`` and
    ``status = "deleted"``. Replay (``collection_session_stats``) then drops them
    from the collection snapshot, and ``latest_md5_by_rel_path`` reports no md5 so
    a later re-upload of the same file is re-indexed instead of skipped.
    """
    sources = [s for s in dict.fromkeys(rel_paths) if s]
    if not sources:
        return 0
    session_id = add_session(
        "（手动删除）",
        recreate=False,
        documents=0,
        chunks=0,
        status="ok",
        collection=collection,
    )
    add_files(
        session_id,
        [
            {
                "filename": Path(s).name,
                "rel_path": s,
                "status": "deleted",
                "chunk_count": 0,
                "file_size": 0,
                "file_md5": None,
            }
            for s in sources
        ],
    )
    return session_id


def collection_session_stats(collection: str | None = None) -> dict[int, dict]:
    """Per-session incremental counts and the collection snapshot (number of
    distinct indexed files/chunks) as of the end of each session. Sessions are
    replayed in file-insertion order; deleted rows (file_md5 IS NULL) are
    excluded from totals. Sessions without files carry the prior snapshot."""
    init_db()
    collection = collection or settings.qdrant_collection
    with _connection() as conn:
        session_ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM sessions WHERE collection = ? ORDER BY id",
                (collection,),
            ).fetchall()
        ]
        files = conn.execute(
            """
            SELECT f.session_id AS sid, f.rel_path, f.file_md5, f.status,
                   f.chunk_count
            FROM files f
            JOIN sessions s ON s.id = f.session_id
            WHERE s.collection = ?
            ORDER BY f.id
            """,
            (collection,),
        ).fetchall()

    result: dict[int, dict] = {}
    state: dict[str, tuple] = {}
    i = 0
    n = len(files)
    for sid in session_ids:
        counts = {"added": 0, "updated": 0, "unchanged": 0, "deleted": 0}
        while i < n and files[i]["sid"] == sid:
            f = files[i]
            counts[f["status"]] += 1
            if f["file_md5"]:
                state[f["rel_path"]] = (f["file_md5"], f["chunk_count"])
            else:
                state.pop(f["rel_path"], None)
            i += 1
        totals = {
            "total_files": len(state),
            "total_chunks": sum(c for _, c in state.values()),
        }
        result[sid] = {**counts, **totals}
    return result


def session_snapshot(
    session_id: int, collection: str | None = None
) -> list[dict]:
    """Snapshot of indexed files in the collection as of the end of session_id."""
    init_db()
    collection = collection or settings.qdrant_collection
    with _connection() as conn:
        target_sid = int(session_id)
        rows = conn.execute(
            """
            SELECT f.rel_path, f.file_md5, f.chunk_count
            FROM files f
            JOIN sessions s ON s.id = f.session_id
            WHERE s.collection = ? AND f.id <= (
                SELECT COALESCE(MAX(id), 0) FROM files WHERE session_id = ?
            )
            ORDER BY f.id
            """,
            (collection, target_sid),
        ).fetchall()
    state: dict[str, dict] = {}
    for r in rows:
        if r["file_md5"]:
            state[r["rel_path"]] = {"rel_path": r["rel_path"], "chunk_count": r["chunk_count"]}
        else:
            state.pop(r["rel_path"], None)
    return sorted(state.values(), key=lambda x: x["rel_path"])
