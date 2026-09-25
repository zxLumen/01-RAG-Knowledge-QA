from __future__ import annotations

import hashlib
import pathlib
import time
import traceback
import uuid
from typing import Callable

from src.config import settings
from src.imports.store import add_files, add_session, latest_md5_by_rel_path, prune_files
from src.ingest import progress as ingest_progress
from src.ingest.chunker import split_documents
from src.ingest.loader import SUPPORTED_EXTENSIONS, load_directory, load_file
from src.qa.llm_config import get_embedding
from src.vectorstore.context import current_collection, reset_collection, set_collection
from src.vectorstore.embedder import get_dense_embeddings, get_sparse_embeddings
from src.vectorstore.store import (
    add_documents,
    copy_source_points,
    create_collection,
    delete_by_source,
    get_client,
    promote_collection,
    source_stats,
    staging_name,
)

# Chunks per embed+upsert flush. Large imports are streamed through this window
# so peak memory does not scale with the collection size.
INGEST_BATCH_SIZE = 128


def _record_session(
    path: str,
    recreate: bool,
    documents: int,
    chunks: int,
    status: str,
    error: str | None,
    started: float,
) -> None:
    add_session(
        path,
        recreate=recreate,
        documents=documents,
        chunks=chunks,
        status=status,
        error=error,
        duration_ms=int((time.monotonic() - started) * 1000),
        embedding_model=get_embedding().get('model') or settings.dense_embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        collection=current_collection(),
    )


def _file_md5(source: str, content: str) -> str:
    p = pathlib.Path(source)
    try:
        return hashlib.md5(p.read_bytes()).hexdigest()
    except OSError:
        return hashlib.md5(content.encode("utf-8")).hexdigest()


def _collect_docs(paths: list[str]) -> list[dict]:
    """Collect documents from dirs/files, deduped by source.

    Returns a list of {"source", "filename", "rel_path", "md5", "docs"} entries.
    """
    loaded: dict[str, dict] = {}
    seen_sources: set[str] = set()
    for raw in paths:
        p = pathlib.Path(raw)
        if p.is_dir():
            entries = load_directory(p)
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            entries = load_file(p)
        else:
            continue
        for doc in entries:
            src = doc.metadata["source"]
            if src in seen_sources:
                continue
            seen_sources.add(src)
            loaded.setdefault(src, {"docs": [], "filename": doc.metadata["filename"]})
            loaded[src]["docs"].append(doc)

    entries = []
    for src, data in loaded.items():
        content = "".join(d.content for d in data["docs"])
        entries.append(
            {
                "source": src,
                "filename": data["filename"],
                "rel_path": src,
                "md5": _file_md5(src, content),
                "docs": data["docs"],
            }
        )
    return sorted(entries, key=lambda e: e["source"])


def _is_within(child: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def _deleted_sources(
    existing: dict[str, dict], current_sources: set[str], paths: list[str]
) -> list[str]:
    import_roots = [pathlib.Path(r).resolve() for r in paths]
    cwd = pathlib.Path.cwd()
    deleted = []
    for rel_path, info in existing.items():
        if rel_path in current_sources:
            continue
        if not info.get("file_md5"):
            continue
        candidate = pathlib.Path(rel_path)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        if not any(_is_within(candidate, r) for r in import_roots):
            continue
        if candidate.exists():
            continue
        deleted.append(rel_path)
    return deleted


def ingest_paths(
    paths: list[str],
    recreate: bool = False,
    delete_missing: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
    collection: str | None = None,
) -> dict:
    token = set_collection(collection) if collection else None
    try:
        return _ingest_paths(paths, recreate, delete_missing, progress)
    finally:
        if token is not None:
            reset_collection(token)


def _deleted_row(src: str) -> dict:
    return {
        "filename": pathlib.Path(src).name,
        "rel_path": src,
        "status": "deleted",
        "chunk_count": 0,
        "file_size": 0,
        "file_md5": None,
    }


def _ingest_paths(
    paths: list[str],
    recreate: bool = False,
    delete_missing: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    started = time.monotonic()
    client = get_client()
    logical = current_collection()

    if progress:
        progress("read", 0, 0)

    existing = {} if recreate else latest_md5_by_rel_path(logical)
    existing_md5 = {k: v.get("file_md5") for k, v in existing.items()}

    entries = _collect_docs(paths)
    current_sources = {e["source"] for e in entries}

    if progress:
        progress("chunk", 0, 0)

    deleted = []
    if delete_missing and existing and not recreate:
        deleted = _deleted_sources(existing, current_sources, paths)

    if not entries:
        # No supported documents under the given paths: never touch the
        # collection, only record tombstones for sources that disappeared.
        if progress:
            progress("done", 0, 0)
        files = []
        for src in deleted:
            delete_by_source(client, src, collection=logical)
            files.append(_deleted_row(src))
        if files:
            session_id = add_session(
                ", ".join(paths),
                recreate=recreate,
                documents=0,
                chunks=0,
                status="ok",
                duration_ms=int((time.monotonic() - started) * 1000),
                embedding_model=get_embedding().get('model') or settings.dense_embedding_model,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                collection=logical,
            )
            add_files(session_id, files)
            return {
                "documents": 0,
                "chunks": 0,
                "added": 0,
                "updated": 0,
                "unchanged": 0,
                "deleted": len(deleted),
            }
        _record_session(
            ", ".join(paths),
            recreate,
            0,
            0,
            "error",
            "No supported documents found",
            started,
        )
        if progress:
            progress("done", 0, 0)
        return {"error": "No supported documents found", "documents": 0, "chunks": 0}

    to_process = [e for e in entries if recreate or existing_md5.get(e["source"]) != e["md5"]]
    changed_sources = {e["source"] for e in to_process}
    deleted_set = set(deleted)

    if not recreate and not to_process and not deleted:
        # Nothing changed: record an unchanged session without rebuilding/rewriting.
        files = [
            {
                "filename": e["filename"],
                "rel_path": e["source"],
                "status": "unchanged",
                "chunk_count": existing.get(e["source"], {}).get("chunk_count", 0),
                "file_size": len(
                    "".join(d.content for d in e["docs"]).encode("utf-8")
                ),
                "file_md5": e["md5"],
            }
            for e in entries
        ]
        session_id = add_session(
            ", ".join(paths),
            recreate=False,
            documents=len(entries),
            chunks=0,
            status="ok",
            duration_ms=int((time.monotonic() - started) * 1000),
            embedding_model=get_embedding().get('model') or settings.dense_embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            collection=logical,
        )
        add_files(session_id, files)
        if progress:
            progress("done", 1, 1)
        return {
            "documents": len(entries),
            "chunks": 0,
            "added": 0,
            "updated": 0,
            "unchanged": len(entries),
            "deleted": 0,
        }

    # Sources already indexed that this run neither re-embeds nor deletes must
    # be carried into the fresh build untouched.
    carry_sources: list[str] = []
    if not recreate:
        indexed = {s["source"] for s in source_stats(client, collection=logical)}
        carry_sources = [
            s for s in indexed if s not in changed_sources and s not in deleted_set
        ]

    chunks = _chunks_for(to_process)
    chunk_counts: dict[str, int] = {}
    for c in chunks:
        src = c["metadata"]["source"]
        chunk_counts[src] = chunk_counts.get(src, 0) + 1

    staging = staging_name(logical)
    create_collection(client, staging)
    promoted = False
    try:
        for src in carry_sources:
            copy_source_points(client, logical, staging, src)

        embed_total = len(chunks) * 2
        if progress:
            progress("embed", 0, embed_total)
        _embed_into(client, staging, chunks, progress, embed_total=embed_total)

        promote_collection(client, logical, staging)
        promoted = True
    except Exception as e:
        traceback.print_exc()
        cancelled = ingest_progress.is_cancelled() or "cancelled" in str(e)
        phase = "cancelled" if cancelled else "error"
        if not promoted:
            try:
                client.delete_collection(staging)
            except Exception:
                pass
        _record_session(
            ", ".join(paths),
            recreate,
            len(entries),
            0,
            phase,
            "cancelled" if cancelled else str(e),
            started,
        )
        if progress:
            progress("done", 0, 0)
        return {"error": phase, "documents": len(entries), "chunks": 0}

    files = []
    added = updated = unchanged_count = 0
    for e in entries:
        src = e["source"]
        chunk_count = chunk_counts.get(src, 0)
        if recreate or existing_md5.get(src) != e["md5"]:
            if existing_md5.get(src) is None:
                status = "added"
                added += 1
            else:
                status = "updated"
                updated += 1
        else:
            status = "unchanged"
            unchanged_count += 1
            chunk_count = existing[src].get("chunk_count", chunk_count)
        files.append(
            {
                "filename": e["filename"],
                "rel_path": src,
                "status": status,
                "chunk_count": chunk_count,
                "file_size": len(
                    "".join(d.content for d in e["docs"]).encode("utf-8")
                ),
                "file_md5": e["md5"],
            }
        )

    for src in deleted:
        files.append(_deleted_row(src))

    if recreate:
        prune_files(logical, {e["source"] for e in entries})

    session_id = add_session(
        ", ".join(paths),
        recreate=recreate,
        documents=len(entries),
        chunks=len(chunks),
        status="ok",
        duration_ms=int((time.monotonic() - started) * 1000),
        embedding_model=get_embedding().get('model') or settings.dense_embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        collection=logical,
    )
    add_files(session_id, files)

    if progress:
        progress("done", 1, 1)

    return {
        "documents": len(entries),
        "chunks": len(chunks),
        "added": added,
        "updated": updated,
        "unchanged": unchanged_count,
        "deleted": len(deleted),
    }


def _chunks_for(to_process: list[dict]) -> list[dict]:
    process_docs = [d for e in to_process for d in e["docs"]]
    chunks = split_documents(process_docs) if process_docs else []
    for c in chunks:
        c["metadata"]["rel_path"] = c["metadata"].get("source")
    return chunks


def _embed_into(
    client,
    collection: str,
    chunks: list[dict],
    progress: Callable[[str, int, int], None] | None,
    *,
    embed_total: int,
) -> None:
    """Embed chunks in windows and upsert each window into ``collection``.

    Only one window of vectors is alive at a time, so peak memory stays flat
    regardless of how many chunks an import produces.
    """
    if not chunks:
        return
    dense_embeddings = get_dense_embeddings()
    try:
        sparse_embeddings = get_sparse_embeddings()
    except Exception:
        sparse_embeddings = None

    for start in range(0, len(chunks), INGEST_BATCH_SIZE):
        if ingest_progress.is_cancelled():
            raise RuntimeError("cancelled")
        window = chunks[start : start + INGEST_BATCH_SIZE]
        texts = [c["text"] for c in window]
        metadatas = [c["metadata"] for c in window]
        ids = [str(uuid.uuid4()) for _ in window]

        dense_vecs = dense_embeddings.embed_documents(texts)
        if progress:
            progress("embed", min(embed_total, (start + len(texts)) * 2), embed_total)

        sparse_vecs: list[dict] = []
        if sparse_embeddings is not None:
            try:
                sparse_results = sparse_embeddings.embed_documents(texts)
                sparse_vecs = [
                    {"indices": s.indices, "values": s.values} for s in sparse_results
                ]
            except Exception:
                if ingest_progress.is_cancelled():
                    raise
        if progress:
            progress("embed", min(embed_total, (start + len(texts)) * 2), embed_total)

        if progress:
            progress("upsert", 0, 0)
        add_documents(
            client,
            texts,
            metadatas,
            ids,
            dense_vecs,
            sparse_vecs,
            collection=collection,
        )


def ingest_path(path: str, recreate: bool = False, delete_missing: bool = True) -> dict:
    return ingest_paths([path], recreate=recreate, delete_missing=delete_missing)
