from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api import ui_config, visitor
from src.api.routes import router

app = FastAPI(title="RAG Knowledge QA", version="1.0.0")

static_dir = Path(__file__).parent.parent.parent / "static"
_index_path = static_dir / "index.html"


@app.on_event("startup")
async def _startup() -> None:
    visitor.ensure_samples()


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
