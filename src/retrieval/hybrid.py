from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.retrieval import config as retrieval_config
from src.vectorstore.context import current_collection
from src.vectorstore.embedder import get_dense_embeddings, get_sparse_embeddings
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


def _item(point) -> dict:
    payload = point.payload or {}
    metadata = payload.get("metadata", {}) or {}
    return {
        "id": point.id,
        "text": payload.get("text", ""),
        "score": point.score,
        "source": metadata.get("source", ""),
        "metadata": metadata,
    }


def _candidate_pool_size(top_k: int, cfg: dict) -> int:
    return max(top_k * int(cfg["candidate_multiplier"]), int(cfg["candidate_min"]))


def _collection_has_sparse(client, collection: str) -> bool:
    """True only when the collection is configured with a ``sparse`` vector.

    Collections imported before sparse support exist but carry dense-only
    points, so a sparse prefetch would return nothing; we fall back to the
    keyword channel for those.
    """
    try:
        params = client.get_collection(collection).config.params
        return bool(getattr(params, "sparse_vectors", None))
    except Exception:
        return False


def _dense_channel(client, collection, query_dense, limit, min_score) -> list[dict]:
    kwargs = {}
    if min_score and min_score > 0:
        kwargs["score_threshold"] = float(min_score)
    res = client.query_points(
        collection_name=collection,
        query=query_dense,
        using="dense",
        with_payload=True,
        limit=limit,
        **kwargs,
    )
    return [_item(p) for p in res.points]


def _sparse_channel(client, collection, query: str, limit) -> list[dict]:
    from qdrant_client.models import SparseVector

    sparse = get_sparse_embeddings().embed_query(query)
    res = client.query_points(
        collection_name=collection,
        query=SparseVector(indices=sparse.indices, values=sparse.values),
        using="sparse",
        with_payload=True,
        limit=limit,
    )
    return [_item(p) for p in res.points]


def _keyword_channel(client, collection, query: str, query_dense, limit) -> list[dict]:
    """Dense query restricted to chunks matching any query keyword.

    This is the legacy lexical signal, kept as an extra RRF channel so old
    (dense-only) collections still get some exact-term recall.
    """
    keywords = _extract_keywords(query)[: int(retrieval_config.get("keyword_limit"))]
    if not keywords:
        return []
    from qdrant_client.models import FieldCondition, Filter, MatchText

    should = [FieldCondition(key="text", match=MatchText(text=kw)) for kw in keywords]
    res = client.query_points(
        collection_name=collection,
        query=query_dense,
        using="dense",
        query_filter=Filter(should=should),
        with_payload=True,
        limit=limit,
    )
    return [_item(p) for p in res.points]


def _rrf_fuse(channels: list[list[dict]], k: int, weights: list[float]) -> list[dict]:
    """Weighted Reciprocal Rank Fusion over several ranked channels."""
    scores: dict[object, float] = {}
    items: dict[object, dict] = {}
    for weight, ranked in zip(weights, channels):
        if weight <= 0:
            continue
        for rank, item in enumerate(ranked):
            key = item["id"]
            items.setdefault(key, item)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1)
    fused = [{**items[key], "score": score} for key, score in scores.items()]
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


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
    query: str,
    top_k: int | None = None,
    collection_name: str | None = None,
    chunks_per_file: int | None = None,
) -> list[RetrievedChunk]:
    cfg = retrieval_config.load()
    top_k = int(top_k or cfg["top_k"])
    top_k = max(1, min(100, top_k))
    chunks_per_file = int(chunks_per_file or cfg["chunks_per_file"])
    collection = collection_name or current_collection()
    client = get_client()

    # A missing collection (e.g. the admin's default before any import) should
    # read as "no relevant documents", not raise a retrieval error.
    if not client.collection_exists(collection):
        return []

    query_dense = get_dense_embeddings().embed_query(query)
    candidate_n = _candidate_pool_size(top_k, cfg)

    try:
        dense_items = _dense_channel(
            client, collection, query_dense, candidate_n, cfg.get("min_score", 0.0)
        )
    except Exception:
        logger.exception("dense recall failed (collection=%s)", collection)
        return []

    channels: list[list[dict]] = [dense_items]
    weights: list[float] = [float(cfg["dense_weight"])]

    if cfg["hybrid_enabled"] and _collection_has_sparse(client, collection):
        try:
            sparse_items = _sparse_channel(client, collection, query, candidate_n)
            if sparse_items:
                channels.append(sparse_items)
                weights.append(float(cfg["sparse_weight"]))
        except Exception:
            logger.exception("sparse recall failed (collection=%s)", collection)

    if int(cfg["keyword_limit"]) > 0:
        try:
            keyword_items = _keyword_channel(
                client, collection, query, query_dense, candidate_n
            )
            if keyword_items:
                channels.append(keyword_items)
                weights.append(float(cfg["keyword_boost"]))
        except Exception:
            logger.exception("keyword recall failed (collection=%s)", collection)

    if len(channels) == 1:
        items = dense_items
    else:
        items = _rrf_fuse(channels, int(cfg["rrf_k"]), weights)

    items = _select_final_chunks(items, top_k, chunks_per_file)
    rerank_top_k = int(cfg["rerank_top_k"])
    if rerank_top_k > 0:
        items = items[:rerank_top_k]

    return [
        RetrievedChunk(text=item["text"], score=item["score"], metadata=item["metadata"])
        for item in items
    ]
