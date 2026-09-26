"""Visitor identity, per-visitor data directories and quotas.

Public demos isolate each visitor behind a cookie: their uploads live under
``data/visitors/<id>/`` and their vectors go into a dedicated Qdrant collection
``visitor_<id>``. Read-only sample documents under ``data/samples/`` are shared
by everyone. Visitors expire after ``VISITOR_TTL_DAYS`` of inactivity and the
global storage is capped at ``GLOBAL_QUOTA`` (oldest visitors are evicted first).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import threading
import time
from pathlib import Path

from src.config import settings

logger = logging.getLogger("rag")

VISITOR_COOKIE = "rag_visitor"
VISITOR_TTL_DAYS = 7
VISITOR_TTL_SECONDS = VISITOR_TTL_DAYS * 24 * 3600

# Limits (bytes)
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = 2 * 1024 * 1024
VISITOR_QUOTA_BYTES = 2 * 1024 * 1024
GLOBAL_QUOTA_BYTES = 2 * 1024 * 1024 * 1024

DATA_DIR = Path("data")
SAMPLES_DIR = DATA_DIR / "samples"
VISITORS_DIR = DATA_DIR / "visitors"
STATE_PATH = Path("qdrant_data/visitors.json")
MINT_STATE_PATH = Path("qdrant_data/mint_hits.json")
MINT_WINDOW_SECONDS = 3600

_lock = threading.RLock()
_state: dict | None = None


def _load_state() -> dict:
    global _state
    if _state is not None:
        return _state
    try:
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    raw.setdefault("visitors", {})
    raw.setdefault("views", {})
    _state = raw
    return _state


def _save_state() -> None:
    if _state is None:
        return
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(
            json.dumps(_state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def new_visitor_id() -> str:
    return secrets.token_hex(8)


def allow_new_identity(ip: str) -> bool:
    """Rate-limit how many brand-new identities one IP may mint per hour.

    Persisted to disk on purpose: an in-memory counter is wiped by every
    redeploy, which would let a caller reset the limit just by waiting for a
    restart. Fails open if the file cannot be written — a rate limiter must
    never be the reason the site stops serving.
    """
    limit = settings.visitor_mint_per_hour
    if limit <= 0:
        return True
    now = time.time()
    with _lock:
        try:
            raw = json.loads(MINT_STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        live: dict[str, list[float]] = {}
        for key, stamps in raw.items():
            kept = []
            for t in stamps if isinstance(stamps, list) else []:
                try:
                    ts = float(t)
                except (TypeError, ValueError):
                    continue
                if now - ts < MINT_WINDOW_SECONDS:
                    kept.append(ts)
            if kept:
                live[str(key)] = kept
        hits = live.get(ip, [])
        allowed = len(hits) < limit
        if allowed:
            hits = [*hits, now]
            live[ip] = hits
        try:
            MINT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = MINT_STATE_PATH.with_name(MINT_STATE_PATH.name + ".tmp")
            tmp.write_text(json.dumps(live), encoding="utf-8")
            tmp.replace(MINT_STATE_PATH)
        except OSError:
            logger.warning("mint rate-limit state not persisted: %s", MINT_STATE_PATH)
            return True
    return allowed


def is_valid_id(visitor_id: str | None) -> bool:
    if not visitor_id:
        return False
    return len(visitor_id) == 16 and all(c in "0123456789abcdef" for c in visitor_id.lower())


def visitor_dir(visitor_id: str) -> Path:
    if not is_valid_id(visitor_id):
        raise ValueError("invalid visitor id")
    return VISITORS_DIR / visitor_id.lower()


def collection_name(visitor_id: str) -> str:
    if not is_valid_id(visitor_id):
        raise ValueError("invalid visitor id")
    return f"visitor_{visitor_id.lower()}"


def visitor_ids_by_age() -> list[tuple[str, float]]:
    """Valid visitor ids paired with their last-seen time, oldest first."""
    with _lock:
        state = _load_state()
        items = [
            (vid, float(ts))
            for vid, ts in state.get("visitors", {}).items()
            if is_valid_id(vid)
        ]
    items.sort(key=lambda x: x[1])
    return items


def visitor_points_total(client=None) -> int:
    """Total points across *visitor* collections only (admin excluded)."""
    from src.vectorstore.store import collection_points, get_client, list_collections

    client = client or get_client()
    total = 0
    for name in list_collections(client):
        if name.startswith("visitor_") and is_valid_id(name[len("visitor_"):]):
            total += collection_points(client, name)
    return total


def ensure_global_capacity(
    exclude_vid: str | None, need_points: int, budget: int, client=None
) -> list[str]:
    """Evict least-recently-active visitors' knowledge bases until ``need_points``
    fits within ``budget``.

    Only visitor collections and their import ledger are removed; uploads,
    chats and — above all — admin collections are never touched. Raises
    ``QuotaExceededError`` when even after eviction it still does not fit.
    """
    from src.imports.store import delete_collection_sessions
    from src.vectorstore.store import collection_points, delete_collection, get_client

    client = client or get_client()
    exclude = (exclude_vid or "").lower()
    candidates: list[tuple[float, str, str, int]] = []
    others_total = 0
    for vid, ts in visitor_ids_by_age():
        if vid == exclude:
            continue
        name = collection_name(vid)  # only ever visitor_<valid-id>
        pts = collection_points(client, name)
        if pts <= 0:
            continue
        candidates.append((ts, vid, name, pts))
        others_total += pts

    evicted: list[str] = []
    for _ts, vid, name, pts in candidates:
        if others_total + need_points <= budget:
            break
        try:
            delete_collection(client, name)
            delete_collection_sessions(name)
        except Exception:
            logger.exception("failed to evict visitor collection %s", name)
            continue
        others_total -= pts
        evicted.append(vid)
    if evicted:
        logger.info("evicted %d visitor collection(s): %s", len(evicted), evicted)

    if others_total + need_points > budget:
        from src.ingest.pipeline import QuotaExceededError

        raise QuotaExceededError(f"全局知识库容量已满（上限 {budget} 分块），请稍后再试")
    return evicted


def get_view(visitor_id: str) -> str | None:
    """The shared collection display name this visitor is currently viewing."""
    if not is_valid_id(visitor_id):
        return None
    with _lock:
        state = _load_state()
        value = state.get("views", {}).get(visitor_id.lower())
    return value if isinstance(value, str) and value else None


def set_view(visitor_id: str, name: str | None) -> None:
    """Persist a visitor's viewed collection (None/empty resets to own)."""
    if not is_valid_id(visitor_id):
        return
    with _lock:
        state = _load_state()
        views = state.setdefault("views", {})
        if name:
            views[visitor_id.lower()] = name
        else:
            views.pop(visitor_id.lower(), None)
        _save_state()


def touch(visitor_id: str) -> None:
    if not is_valid_id(visitor_id):
        return
    with _lock:
        state = _load_state()
        state["visitors"][visitor_id.lower()] = time.time()
        _save_state()


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def visitor_usage(visitor_id: str) -> int:
    try:
        d = visitor_dir(visitor_id)
    except ValueError:
        return 0
    return _dir_size(d) if d.is_dir() else 0


def global_usage() -> int:
    if not VISITORS_DIR.is_dir():
        return 0
    return _dir_size(VISITORS_DIR)


def _delete_collection(visitor_id: str) -> None:
    try:
        from src.vectorstore.store import delete_collection, get_client

        name = collection_name(visitor_id)
        client = get_client()
        if client.collection_exists(name):
            delete_collection(client, name)
    except Exception:
        pass


def remove_visitor(visitor_id: str) -> None:
    """Delete a visitor's files, collection and state entry."""
    vid = (visitor_id or "").lower()
    if not is_valid_id(vid):
        return
    with _lock:
        try:
            shutil.rmtree(visitor_dir(vid), ignore_errors=True)
        except (OSError, ValueError):
            pass
        state = _load_state()
        state["visitors"].pop(vid, None)
        state.get("views", {}).pop(vid, None)
        _save_state()
    try:
        from src.chats import store as chat_store
        chat_store.purge_owner(collection_name(vid))
    except Exception:
        pass
    _delete_collection(vid)


def cleanup_expired() -> None:
    """Drop visitors idle longer than the TTL (lazy, called per request)."""
    now = time.time()
    with _lock:
        state = _load_state()
        expired = [
            vid
            for vid, ts in state["visitors"].items()
            if now - float(ts) > VISITOR_TTL_SECONDS
        ]
    for vid in expired:
        remove_visitor(vid)


def enforce_global_quota(incoming: int = 0) -> None:
    """Evict oldest visitors until global usage leaves room for ``incoming``."""
    if global_usage() + incoming <= GLOBAL_QUOTA_BYTES:
        return
    with _lock:
        state = _load_state()
        ordered = sorted(state["visitors"].items(), key=lambda kv: float(kv[1]))
    for vid, _ts in ordered:
        if global_usage() + incoming <= GLOBAL_QUOTA_BYTES:
            break
        remove_visitor(vid)


def ensure_samples() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
