from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api import visitor
from src.api.routes import router

app = FastAPI(title="RAG Knowledge QA", version="1.0.0")


@app.on_event("startup")
async def _startup() -> None:
    visitor.ensure_samples()


app.include_router(router)

static_dir = Path(__file__).parent.parent.parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
