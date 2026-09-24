from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from src.api import admin_auth, visitor
from src.api.schemas import (
    CollectionRenameRequest,
    CollectionsResponse,
    CollectionSwitchRequest,
    ConfigResponse,
    ImportFile,
    ImportListResponse,
    ImportSession,
    ImportSessionDetail,
    IngestCancelRequest,
    IngestRequest,
    IngestResponse,
    LLMActiveRequest,
    LLMConfigResponse,
    LLMEmbeddingRequest,
    LLMProfileRequest,
    LLMTestRequest,
    ModelInfo,
    ModelSelectRequest,
    ModelsResponse,
    PasswordChangeRequest,
    ProvidersResponse,
    QueryCancelRequest,
    QueryRequest,
    StatusResponse,
)
from src.config import settings
from src.imports.store import (
    collection_session_stats,
    delete_collection_sessions,
    get_session,
    list_sessions,
    session_snapshot,
)
from src.imports.store import list_files as list_session_files
from src.ingest import progress as ingest_progress
from src.ingest.loader import load_file
from src.ingest.pipeline import ingest_paths
from src.qa import cancel as generation_cancel
from src.qa import llm as llm_client
from src.qa import llm_config, providers
from src.qa.chain import answer_question_stream
from src.vectorstore.naming import (
    delete_alias,
    display_name,
    rename_display,
    storage_for_display,
)
from src.vectorstore.store import (
    collection_info,
    delete_collection,
    get_client,
    list_collections,
    set_active_collection,
    source_stats,
)

router = APIRouter(prefix="/api")
logger = logging.getLogger("rag")


def require_admin(
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
) -> None:
    """Guard write operations when an admin password is configured."""
    if not admin_auth.password_set():
        return
    provided = x_admin_token or ""
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    if not admin_auth.verify(provided):
        raise HTTPException(status_code=401, detail="需要管理员令牌（ADMIN_TOKEN）")


def resolve_visitor(request: Request, response: Response) -> str:
    """Return the visitor id from the cookie, minting one if absent.

    Also lazily evicts expired visitors and enforces the global quota.
    """
    vid = request.cookies.get(visitor.VISITOR_COOKIE)
    if not visitor.is_valid_id(vid):
        vid = visitor.new_visitor_id()
        response.set_cookie(
            visitor.VISITOR_COOKIE,
            vid,
            max_age=visitor.VISITOR_TTL_SECONDS,
            httponly=True,
            samesite="lax",
        )
    visitor.cleanup_expired()
    visitor.enforce_global_quota()
    visitor.touch(vid)
    return vid


@router.post("/admin/verify")
async def admin_verify(_admin: None = Depends(require_admin)):
    if not admin_auth.password_set():
        raise HTTPException(status_code=400, detail="服务端未设置 ADMIN_TOKEN")
    return {"ok": True}


@router.post("/admin/password")
async def admin_change_password(
    req: PasswordChangeRequest, _admin: None = Depends(require_admin)
):
    try:
        admin_auth.change_password(req.old_password, req.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


_rate_lock = threading.Lock()
_rate_hits: dict[str, list[float]] = {}


def enforce_rate_limit(request: Request) -> None:
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    ip = request.client.host if request.client else "?"
    now = time.time()
    with _rate_lock:
        hits = [t for t in _rate_hits.get(ip, []) if now - t < 60]
        if len(hits) >= limit:
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
        hits.append(now)
        _rate_hits[ip] = hits


def _within(child: Path, root: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def _visitor_roots(visitor_id: str) -> list[Path]:
    cwd = Path.cwd().resolve()
    roots = [(cwd / visitor.SAMPLES_DIR).resolve()]
    try:
        roots.append(visitor.visitor_dir(visitor_id).resolve())
    except ValueError:
        pass
    return roots


def _is_admin(request: Request) -> bool:
    """True when a valid admin password is presented (and one is configured)."""
    if not admin_auth.password_set():
        return False
    provided = request.headers.get("x-admin-token", "")
    if not provided:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
    return admin_auth.verify(provided)


def visitor_collection(request: Request, response: Response) -> str | None:
    """Effective collection for this request.

    Returns the visitor's dedicated collection in demo mode, otherwise None
    (meaning the admin/default collection). A valid admin token bypasses visitor
    scoping so the operator can see the shared/admin knowledge bases.
    """
    if not settings.demo_mode or _is_admin(request):
        return None
    vid = resolve_visitor(request, response)
    return visitor.collection_name(vid)


@router.post("/query")
async def query(
    req: QueryRequest,
    request: Request,
    response: Response,
    _rate: None = Depends(enforce_rate_limit),
):
    visitor_scope = visitor_collection(request, response)
    if visitor_scope:
        # Visitor: always their own collection; ignore any client-supplied name.
        collection = visitor_scope
    elif req.collection:
        collection = storage_for_display(req.collection)
    else:
        collection = settings.qdrant_collection

    def event_stream():
        for event in answer_question_stream(
            req.question,
            top_k=req.top_k,
            collection_name=collection,
            gen_id=req.session_id,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/query/cancel")
async def cancel_query(req: QueryCancelRequest):
    cancelled = generation_cancel.request_cancel(req.session_id)
    return {"cancelled": cancelled}


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, request: Request, response: Response):
    if ingest_progress.snapshot()["running"]:
        return IngestResponse(
            status="busy",
            documents=0,
            chunks=0,
            error="已有导入正在进行中，请等待完成或先取消",
        )
    if req.paths:
        for_ingest = list(dict.fromkeys(req.paths))
    else:
        for_ingest = [req.path]
    if settings.demo_mode and not _is_admin(request):
        vid = resolve_visitor(request, response)
        allowed = _visitor_roots(vid)
        clean = []
        for path in for_ingest:
            resolved = (Path.cwd() / path).resolve()
            if any(_within(resolved, root) for root in allowed):
                clean.append(path)
        for_ingest = clean
        if not for_ingest:
            return IngestResponse(
                status="error", documents=0, chunks=0,
                error="只能导入样例或你自己上传的文件",
            )
    ingest_progress.begin()
    try:
        result = await run_in_threadpool(
            ingest_paths,
            for_ingest,
            recreate=req.recreate,
            delete_missing=req.delete_missing,
            progress=ingest_progress.set_phase,
            collection=visitor_collection(request, response),
        )
    except Exception as exc:
        cancelled = ingest_progress.is_cancelled()
        ingest_progress.finish(
            "cancelled" if cancelled else "error", 0, 0,
            summary=None if cancelled else {"error": str(exc)},
        )
        if cancelled:
            return IngestResponse(status="cancelled", documents=0, chunks=0, error="cancelled")
        raise
    if "error" in result:
        cancelled = ingest_progress.is_cancelled() or result.get("error") == "cancelled"
        ingest_progress.finish(
            "cancelled" if cancelled else "error", 0, 0,
            summary={"error": result["error"]},
        )
        return IngestResponse(
            status="cancelled" if cancelled else "error",
            documents=0,
            chunks=0,
            error=result["error"],
        )
    ingest_progress.finish(
        "done", 1, 1,
        summary={
            "documents": result.get("documents", 0),
            "chunks": result.get("chunks", 0),
            "added": result.get("added", 0),
            "updated": result.get("updated", 0),
            "unchanged": result.get("unchanged", 0),
            "deleted": result.get("deleted", 0),
        },
    )
    return IngestResponse(
        status="ok",
        documents=result.get("documents", 0),
        chunks=result.get("chunks", 0),
        added=result.get("added", 0),
        updated=result.get("updated", 0),
        unchanged=result.get("unchanged", 0),
        deleted=result.get("deleted", 0),
    )


@router.get("/ingest/progress")
async def ingest_progress_endpoint():
    return ingest_progress.snapshot()


@router.post("/ingest/cancel")
async def ingest_cancel(req: Optional[IngestCancelRequest] = None):
    snap = ingest_progress.snapshot()
    logger.info(
        "cancel requested: run_id=%s current_started_at=%s running=%s",
        None if req is None else req.run_id,
        snap.get("started_at"),
        snap.get("running"),
    )
    if snap.get("running") and not ingest_progress.is_current_run(
        None if req is None else req.run_id, snap
    ):
        logger.warning("cancel ignored for stale run_id=%s", None if req is None else req.run_id)
        return {**snap, "ignored_stale_cancel": True}
    ingest_progress.cancel()
    return snap


@router.get("/status", response_model=StatusResponse)
async def status(request: Request, response: Response):
    client = get_client()
    collection = visitor_collection(request, response)
    name = collection or settings.qdrant_collection
    info = collection_info(client, collection=name)
    label = display_name(name) if not collection else "我的知识库"
    if info is None:
        return StatusResponse(collection=label, points_count=None, status="not_found")
    return StatusResponse(
        collection=label,
        points_count=info["points_count"],
        status=str(info["status"]),
    )


@router.get("/config", response_model=ConfigResponse)
async def config():
    active = llm_config.get_active()
    emb = llm_config.get_embedding()
    return ConfigResponse(
        embedding_model=emb.get("model", settings.dense_embedding_model),
        embedding_provider=emb.get("provider", "ollama"),
        embedding_dim=llm_config.embedding_dim(),
        llm_model=active.get("model", ""),
        llm_provider=active.get("provider", ""),
        llm_protocol=active.get("protocol", ""),
        llm_base_url=active.get("base_url", ""),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        qdrant_url=settings.qdrant_url,
    )


# --- LLM model management -------------------------------------------------


@router.get("/llm/providers", response_model=ProvidersResponse)
async def llm_providers():
    return ProvidersResponse(
        providers=providers.list_providers(),
        embedding_dims=providers.KNOWN_EMBEDDING_DIMS,
    )


@router.get("/llm/config", response_model=LLMConfigResponse)
async def llm_config_get():
    return LLMConfigResponse(**llm_config.public_config())


@router.post("/llm/profiles", response_model=LLMConfigResponse)
async def llm_profile_upsert(req: LLMProfileRequest, _admin: None = Depends(require_admin)):
    data = req.model_dump()
    if data.get("temperature") is None:
        data.pop("temperature", None)
    llm_config.upsert_profile(data)
    return LLMConfigResponse(**llm_config.public_config())


@router.delete("/llm/profiles/{profile_id}", response_model=LLMConfigResponse)
async def llm_profile_delete(profile_id: str, _admin: None = Depends(require_admin)):
    if not llm_config.delete_profile(profile_id):
        raise HTTPException(status_code=400, detail="至少保留一个档案，或档案不存在")
    return LLMConfigResponse(**llm_config.public_config())


@router.post("/llm/active", response_model=LLMConfigResponse)
async def llm_set_active(req: LLMActiveRequest, _admin: None = Depends(require_admin)):
    if not llm_config.set_active(req.id):
        raise HTTPException(status_code=404, detail="档案不存在")
    return LLMConfigResponse(**llm_config.public_config())


@router.post("/llm/embedding/profiles", response_model=LLMConfigResponse)
async def llm_embedding_upsert(req: LLMEmbeddingRequest, _admin: None = Depends(require_admin)):
    data = req.model_dump()
    if data.get("dim") is None:
        data.pop("dim", None)
    llm_config.upsert_embedding_profile(data)
    return LLMConfigResponse(**llm_config.public_config())


@router.delete("/llm/embedding/profiles/{profile_id}", response_model=LLMConfigResponse)
async def llm_embedding_delete(profile_id: str, _admin: None = Depends(require_admin)):
    if not llm_config.delete_embedding_profile(profile_id):
        raise HTTPException(status_code=400, detail="至少保留一个嵌入档案，或档案不存在")
    return LLMConfigResponse(**llm_config.public_config())


@router.post("/llm/embedding/active", response_model=LLMConfigResponse)
async def llm_embedding_set_active(req: LLMActiveRequest, _admin: None = Depends(require_admin)):
    if not llm_config.set_embedding_active(req.id):
        raise HTTPException(status_code=404, detail="档案不存在")
    return LLMConfigResponse(**llm_config.public_config())


def _friendly_llm_error(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    if "401" in text or "unauthorized" in low or "authorization required" in low:
        return "认证失败：需要有效的 API Key（请填写后再获取）"
    if "403" in text:
        return "无权限：该 Key 不能访问此服务"
    if "404" in text:
        return "接口不存在：请检查 Base URL（通常需以 /v1 结尾）"
    if "timed out" in low or "timeout" in low:
        return "连接超时：请检查网络或 Base URL"
    return f"连接失败: {text}"


@router.post("/llm/test")
async def llm_test(req: LLMTestRequest, _admin: None = Depends(require_admin)):
    if req.id:
        profile = (
            llm_config.get_embedding_profile(req.id)
            if req.target == "embedding"
            else llm_config.get_profile(req.id)
        )
        if profile is None:
            raise HTTPException(status_code=404, detail="档案不存在")
        test_profile = dict(profile)
    elif req.target == "embedding":
        emb = llm_config.get_embedding_active()
        provider = req.provider or emb.get("provider") or "ollama"
        test_profile = {
            "provider": provider,
            "protocol": providers.protocol_for(provider),
            "base_url": (
                req.base_url or emb.get("base_url") or providers.default_base_url(provider)
            ).rstrip("/"),
            "api_key": req.api_key or emb.get("api_key", ""),
            "model": req.model or emb.get("model", ""),
        }
    else:
        provider = req.provider or "custom"
        test_profile = {
            "provider": provider,
            "protocol": providers.protocol_for(provider),
            "base_url": (req.base_url or providers.default_base_url(provider)).rstrip("/"),
            "api_key": req.api_key or "",
            "model": req.model or "",
        }
    if req.api_key:
        test_profile["api_key"] = req.api_key
    try:
        models = await run_in_threadpool(llm_client.list_models, test_profile)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_friendly_llm_error(exc))
    models = providers.filter_by_purpose(models, req.target)
    return {"ok": True, "models": models}


def _list_ollama_models() -> list[dict]:
    import requests as _requests

    active = llm_config.get_active()
    base = active.get("base_url") or settings.ollama_base_url
    resp = _requests.get(f"{base}/api/tags", timeout=10)
    resp.raise_for_status()
    return resp.json().get("models", [])


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    try:
        tags = await run_in_threadpool(_list_ollama_models)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"无法连接模型服务: {exc}")
    models = [
        ModelInfo(name=m.get("name", ""), size=m.get("size"))
        for m in tags
        if m.get("name")
    ]
    models.sort(key=lambda m: m.name)
    return ModelsResponse(models=models, current=llm_config.get_current_model())


@router.post("/models", response_model=ModelsResponse)
async def select_model(req: ModelSelectRequest):
    name = (req.model or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="模型名不能为空")
    llm_config.set_current_model(name)
    try:
        tags = await run_in_threadpool(_list_ollama_models)
    except Exception:
        tags = []
    models = [
        ModelInfo(name=m.get("name", ""), size=m.get("size")) for m in tags if m.get("name")
    ]
    models.sort(key=lambda m: m.name)
    return ModelsResponse(models=models, current=llm_config.get_current_model())


@router.get("/files")
async def list_files(request: Request, response: Response):
    cwd = Path.cwd().resolve()
    collection = visitor_collection(request, response)
    if settings.demo_mode and not _is_admin(request):
        roots = _visitor_roots(
            request.cookies.get(visitor.VISITOR_COOKIE, "")
        )
        sample_root = (cwd / visitor.SAMPLES_DIR).resolve()
    else:
        roots = [(cwd / "data").resolve()]
        sample_root = None

    disk: dict[str, dict] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*")):
            if not f.is_file() or f.name.startswith("."):
                continue
            resolved = f.resolve()
            disk[str(resolved)] = {
                "rel": str(resolved.relative_to(cwd)),
                "name": f.name,
                "sample": bool(sample_root and _within(resolved, sample_root)),
            }

    try:
        stats = source_stats(get_client(), collection=collection)
    except Exception:
        stats = []
    kb = {str(Path(s["source"]).resolve()): s["chunks"] for s in stats}

    items = []
    imported_count = 0
    for abs_key, info in sorted(disk.items()):
        chunks = kb.get(abs_key)
        imported = chunks is not None
        if imported:
            imported_count += 1
        items.append(
            {
                "rel": info["rel"],
                "name": info["name"],
                "sample": info["sample"],
                "imported": imported,
                "chunks": chunks or 0,
            }
        )

    missing = [
        {"filename": Path(s["source"]).name, "chunks": s["chunks"]}
        for s in stats
        if str(Path(s["source"]).resolve()) not in disk
    ]
    return {
        "root": "data",
        "items": items,
        "missing": missing,
        "summary": {
            "total": len(items),
            "imported": imported_count,
            "unimported": len(items) - imported_count,
        },
    }


@router.get("/files/content")
async def file_content(rel: str, request: Request, response: Response):
    cwd = Path.cwd().resolve()
    target = (cwd / rel).resolve()
    if settings.demo_mode and not _is_admin(request):
        vid = request.cookies.get(visitor.VISITOR_COOKIE, "")
        roots = _visitor_roots(vid)
    else:
        roots = [(cwd / "data").resolve()]
    if not any(_within(target, root) for root in roots):
        raise HTTPException(status_code=400, detail="path is outside allowed directories")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    try:
        docs = load_file(target)
        content = "\n\n".join(d.content for d in docs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot read file: {e}")
    return {"rel": rel, "name": target.name, "content": content}


ALLOWED_UPLOAD_EXT = {".md", ".txt", ".pdf"}


@router.post("/upload")
async def upload(
    request: Request,
    response: Response,
    files: list[UploadFile] = File(...),
):
    if not settings.demo_mode:
        raise HTTPException(status_code=400, detail="上传功能仅在演示模式下开启")
    vid = resolve_visitor(request, response)
    dest = visitor.visitor_dir(vid)
    dest.mkdir(parents=True, exist_ok=True)

    total_incoming = 0
    payloads: list[tuple[str, bytes]] = []
    for uf in files:
        name = Path(uf.filename or "").name
        if not name or name.startswith("."):
            raise HTTPException(status_code=400, detail=f"非法文件名: {uf.filename!r}")
        if Path(name).suffix.lower() not in ALLOWED_UPLOAD_EXT:
            raise HTTPException(
                status_code=400, detail=f"不支持的文件类型: {name}（仅 md/txt/pdf）"
            )
        data = await uf.read()
        if len(data) > visitor.MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"{name} 超过单文件上限 2M")
        total_incoming += len(data)
        if total_incoming > visitor.MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="单次上传总量超过 2M")
        payloads.append((name, data))

    used = visitor.visitor_usage(vid)
    if used + total_incoming > visitor.VISITOR_QUOTA_BYTES:
        raise HTTPException(
            status_code=413,
            detail="你的可用空间仅 2M，请先删除部分已上传文件",
        )
    visitor.enforce_global_quota(total_incoming)

    saved = []
    for name, data in payloads:
        target = dest / name
        try:
            target.write_bytes(data)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"保存失败: {e}")
        saved.append(str(target.resolve().relative_to(Path.cwd().resolve())))

    return {"status": "ok", "saved": saved, "usage": visitor.visitor_usage(vid)}


@router.get("/collections", response_model=CollectionsResponse)
async def collections(request: Request, response: Response):
    if settings.demo_mode and not _is_admin(request):
        resolve_visitor(request, response)
        return CollectionsResponse(current="我的知识库", collections=["我的知识库"])
    return CollectionsResponse(
        current=display_name(settings.qdrant_collection),
        collections=[display_name(c) for c in list_collections(get_client())],
    )


@router.post("/collections/switch", response_model=CollectionsResponse)
async def switch_collection(req: CollectionSwitchRequest, request: Request):
    if settings.demo_mode and not _is_admin(request):
        raise HTTPException(status_code=403, detail="访客无权切换集合")
    try:
        set_active_collection(get_client(), req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CollectionsResponse(
        current=display_name(settings.qdrant_collection),
        collections=[display_name(c) for c in list_collections(get_client())],
    )


@router.post("/collections/rename", response_model=CollectionsResponse)
async def rename_collection(req: CollectionRenameRequest, request: Request):
    if settings.demo_mode and not _is_admin(request):
        raise HTTPException(status_code=403, detail="访客无权重命名集合")
    try:
        rename_display(req.name, req.new_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CollectionsResponse(
        current=display_name(settings.qdrant_collection),
        collections=[display_name(c) for c in list_collections(get_client())],
    )


@router.post("/collections/delete", response_model=CollectionsResponse)
async def remove_collection(req: CollectionSwitchRequest, request: Request):
    if settings.demo_mode and not _is_admin(request):
        raise HTTPException(status_code=403, detail="访客无权删除集合")
    client = get_client()
    storage = storage_for_display(req.name)
    existing = list_collections(client)
    if storage not in existing:
        raise HTTPException(status_code=404, detail="集合不存在")
    if len(existing) <= 1:
        raise HTTPException(status_code=400, detail="至少保留一个集合")
    delete_alias(storage)
    delete_collection_sessions(storage)
    delete_collection(client, storage)
    if settings.qdrant_collection == storage:
        settings.qdrant_collection = next(c for c in existing if c != storage)
    return CollectionsResponse(
        current=display_name(settings.qdrant_collection),
        collections=[display_name(c) for c in list_collections(client)],
    )


@router.get("/imports", response_model=ImportListResponse)
async def list_imports(
    request: Request,
    response: Response,
    q: Optional[str] = None,
    limit: int = 100,
    collection: Optional[str] = None,
):
    if settings.demo_mode and not _is_admin(request):
        # Visitor: always their own collection; ignore any client-supplied name.
        vid = request.cookies.get(visitor.VISITOR_COOKIE, "")
        collection = visitor.collection_name(vid) if visitor.is_valid_id(vid) else ""
    else:
        collection = storage_for_display(collection) if collection else settings.qdrant_collection
    sessions = list_sessions(q=q, limit=limit, collection=collection)
    stats = collection_session_stats(collection)

    live = None
    newest_id = max(stats) if stats else None
    if sessions and newest_id is not None and sessions[0]["id"] == newest_id:
        try:
            src = source_stats(get_client(), collection=collection)
            live = (len(src), sum(s["chunks"] for s in src))
        except Exception:
            live = None

    out = []
    for s in sessions:
        st = stats.get(s["id"], {})
        row = {
            **s,
            "recreate": bool(s["recreate"]),
            "added": st.get("added", 0),
            "updated": st.get("updated", 0),
            "unchanged": st.get("unchanged", 0),
            "deleted": st.get("deleted", 0),
            "total_files": st.get("total_files", 0),
            "total_chunks": st.get("total_chunks", 0),
        }
        if live and row["id"] == newest_id:
            row["total_files"], row["total_chunks"] = live
        out.append(ImportSession(**row))
    return ImportListResponse(count=len(out), sessions=out)


@router.get("/imports/{session_id}", response_model=ImportSessionDetail)
async def import_detail(session_id: int):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Import session not found")
    files = [ImportFile(**f) for f in list_session_files(session_id)]
    st = collection_session_stats(session["collection"]).get(session_id, {})
    snapshot = [
        ImportFile(
            filename=Path(f["rel_path"]).name,
            rel_path=f["rel_path"],
            status="present",
            chunk_count=f["chunk_count"],
            file_size=0,
        )
        for f in session_snapshot(session_id, session["collection"])
    ]
    return ImportSessionDetail(
        **{
            **session,
            "recreate": bool(session["recreate"]),
            "added": st.get("added", 0),
            "updated": st.get("updated", 0),
            "unchanged": st.get("unchanged", 0),
            "deleted": st.get("deleted", 0),
            "total_files": st.get("total_files", 0),
            "total_chunks": st.get("total_chunks", 0),
        },
        files=files,
        snapshot_files=snapshot,
    )
