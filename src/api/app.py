from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api import admin_auth, ui_config, visitor
from src.api.routes import router

logger = logging.getLogger("rag")

app = FastAPI(title="RAG Knowledge QA", version="1.0.0")

static_dir = Path(__file__).parent.parent.parent / "static"
_index_path = static_dir / "index.html"


def _ensure_active_collection() -> None:
    """Adopt the persisted active collection and make sure it exists."""
    from src.vectorstore import active
    from src.vectorstore.store import ensure_collection, get_client

    active.apply()
    try:
        ensure_collection(get_client(), recreate=False)
    except Exception:  # noqa: BLE001 - a failure here must not block startup
        logger.exception("failed to ensure the active collection exists")


@app.on_event("startup")
async def _startup() -> None:
    visitor.ensure_samples()
    _ensure_active_collection()
    if admin_auth.stored_password_set():
        logger.info(
            "管理员密码已设置（qdrant_data/admin.json）；如遗忘，可用 ADMIN_TOKEN "
            "调用 POST /api/admin/reset 重置"
        )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve the SPA with the current theme allowlist injected server-side.

    This lets the very first paint use the admin-configured default theme,
    avoiding a flash of the hardcoded fallback theme. ``no-store`` keeps proxies
    and browsers from serving a stale injection after the config changes.
    """
    cfg = ui_config.load()
    payload = json.dumps(
        {"themes": cfg["themes"], "default_theme": cfg["default_theme"]},
        ensure_ascii=False,
    )
    try:
        html = _index_path.read_text(encoding="utf-8")
    except OSError:
        return HTMLResponse(
            "<h1>index.html not found</h1>",
            status_code=500,
            headers={"Cache-Control": "no-store"},
        )
    return HTMLResponse(
        html.replace("__UI_CONFIG__", payload),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/version", include_in_schema=False)
async def version() -> dict:
    """Build identifier, handy for checking whether a running server is current."""
    return {"version": app.version, "build": os.environ.get("APP_BUILD", "dev")}


app.include_router(router)

app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
