"""Per-owner import progress.

Each import is owned by the admin (a single shared operator) or by one visitor
(``v:<id>``). State is keyed by owner so a visitor never sees — or cancels —
someone else's import. Only one import runs at a time globally (the embedded
Qdrant and the embedding window are not safe to run concurrently), but the
"busy" state and progress are only visible to the owner who started it.

The worker thread reports progress through the module functions, which read the
active owner from a contextvar set by the request handler before dispatching to
the thread pool (anyio copies the context into the worker thread).
"""

from __future__ import annotations

import contextvars
import threading
import time
from typing import Any

_lock = threading.Lock()
_runs: dict[str, dict[str, Any]] = {}
_current: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ingest_owner", default=None
)


def _idle() -> dict[str, Any]:
    return {
        "running": False,
        "phase": "",
        "done": 0,
        "total": 0,
        "started_at": 0,
        "summary": None,
    }


def begin(owner: str) -> bool:
    """Start a run for ``owner``. Returns False when one is already running."""
    with _lock:
        if any(r.get("running") for r in _runs.values()):
            return False
        _runs[owner] = {
            **_idle(),
            "running": True,
            "phase": "read",
            "started_at": time.time(),
            "cancelled": False,
        }
    return True


def bind(owner: str) -> contextvars.Token:
    return _current.set(owner)


def unbind(token: contextvars.Token) -> None:
    _current.reset(token)


def _state(owner: str | None = None) -> dict[str, Any] | None:
    with _lock:
        return _runs.get(owner if owner is not None else (_current.get() or ""))


def is_cancelled() -> bool:
    state = _state()
    return bool(state and state.get("cancelled"))


def cancel(owner: str | None = None) -> bool:
    with _lock:
        state = _runs.get(owner if owner is not None else (_current.get() or ""))
        if not state:
            return False
        state["cancelled"] = True
        return True


def set_phase(phase: str, done: int = 0, total: int = 0) -> None:
    with _lock:
        state = _runs.get(_current.get() or "")
        if state is not None:
            state["phase"] = phase
            state["done"] = done
            state["total"] = total


def update(done: int, total: int) -> None:
    with _lock:
        state = _runs.get(_current.get() or "")
        if state is not None:
            state["done"] = done
            state["total"] = total


def finish(
    phase: str = "done", done: int = 1, total: int = 1, summary: Any = None
) -> None:
    with _lock:
        state = _runs.get(_current.get() or "")
        if state is not None:
            state.update(
                running=False, phase=phase, done=done, total=total, summary=summary
            )


def snapshot(owner: str | None = None) -> dict[str, Any]:
    state = _state(owner)
    if state is None:
        return _idle()
    with _lock:
        return dict(state)


def is_current_run(run_id: float | None, snapshot: dict[str, Any] | None = None) -> bool:
    """A cancel request only affects the run it was issued for."""
    if run_id is None:
        return True
    started = (snapshot or {}).get("started_at") or 0
    return abs(float(started) - float(run_id)) < 0.5
