from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.config import settings
from src.vectorstore.context import current_collection
from src.vectorstore.embedder import get_dense_embeddings
from src.vectorstore.store import get_client

logger = logging.getLogger("rag")


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


def _extract_keywords(query: str) -> list[str]:
    keywords = []
    for token in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", query):
        if len(token) >= 2:
            keywords.append(token)
    return keywords


def _build_metadata_filter(keywords: list[str]) -> dict | None:
    if not keywords:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchText

    conditions = []
    for kw in keywords[:5]:
        conditions.append(
            FieldCondition(
                key="text",
                match=MatchText(text=kw),
            )
        )
    return Filter(must=conditions)


def _candidate_pool_size(top_k: int) -> int:
    return max(top_k * 4, 24)


def _select_final_chunks(items: list[dict], top_k: int, chunks_per_file: int) -> list[dict]:
    """Pick chunks guaranteeing at most ``top_k`` distinct files.

    Phase 1: greedy by score chooses the ``top_k`` distinct files (each
    represented by its best-scoring chunk). Phase 2: among candidates that
    belong to the selected files, keep up to ``chunks_per_file`` per file,
    sorted by descending score.
    """
    items = sorted(items, key=lambda x: x["score"], reverse=True)

    files: dict[str, dict] = {}
    for item in items:
        source = item.get("source", "")
        if source and source not in files:
            files[source] = item
        if len(files) >= top_k:
            break

    selected = set(files.keys())

    by_file: dict[str, list[dict]] = {source: [] for source in selected}
    for item in items:
        source = item.get("source", "")
        if source in by_file and len(by_file[source]) < chunks_per_file:
            by_file[source].append(item)

    final = []
    for bucket in by_file.values():
        final.extend(bucket)
    return sorted(final, key=lambda x: x["score"], reverse=True)


def search(
    query: str, top_k: int | None = None, collection_name: str | None = None
) -> list[RetrievedChunk]:
    top_k = top_k or settings.top_k
    chunks_per_file = settings.chunks_per_file
    collection = collection_name or current_collection()
    client = get_client()

    dense_embeddings = get_dense_embeddings()
    query_dense = dense_embeddings.embed_query(query)

    candidate_n = _candidate_pool_size(top_k)

    dense_results = client.query_points(
        collection_name=collection,
        query=query_dense,
        using="dense",
        with_payload=True,
        limit=candidate_n,
    )

    dense_items = []
    for point in dense_results.points:
        payload = point.payload or {}
        dense_items.append({
            "id": point.id,
            "text": payload.get("text", ""),
            "score": point.score,
            "source": (payload.get("metadata", {}) or {}).get("source", ""),
            "metadata": payload.get("metadata", {}),
        })

    keywords = _extract_keywords(query)
    if keywords:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchText

            should_conditions = []
            for kw in keywords[:5]:
                should_conditions.append(
                    FieldCondition(
                        key="text",
                        match=MatchText(text=kw),
                    )
                )
            keyword_filter = Filter(should=should_conditions)

            keyword_results = client.query_points(
                collection_name=collection,
                query=query_dense,
                using="dense",
                query_filter=keyword_filter,
                with_payload=True,
                limit=candidate_n,
            )

            seen_ids = {item["id"] for item in dense_items}
            for point in keyword_results.points:
                if point.id not in seen_ids:
                    payload = point.payload or {}
                    dense_items.append({
                        "id": point.id,
                        "text": payload.get("text", ""),
                        "score": point.score * 1.1,
                        "source": (payload.get("metadata", {}) or {}).get("source", ""),
                        "metadata": payload.get("metadata", {}),
                    })
                    seen_ids.add(point.id)
        except Exception:
            logger.exception("keyword recall failed (collection=%s)", collection)

    items: list[dict] = _select_final_chunks(dense_items, top_k, chunks_per_file)

    chunks: list[RetrievedChunk] = []
    for item in items:
        chunks.append(
            RetrievedChunk(
                text=item["text"],
                score=item["score"],
                metadata=item["metadata"],
            )
        )
    return chunks
