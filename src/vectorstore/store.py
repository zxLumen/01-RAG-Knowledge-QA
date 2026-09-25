from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from src.config import settings
from src.qa.llm_config import embedding_dim
from src.vectorstore import active
from src.vectorstore.context import current_collection
from src.vectorstore.naming import (
    drop_creation,
    order_index,
    record_creation,
    register_alias,
    storage_name,
)

_client: QdrantClient | None = None
_write_lock = threading.RLock()

# A batched import is built into a throwaway collection first and then promoted
# to the logical name via a Qdrant alias. The physical name carries this marker
# so orphaned staging collections can be reclaimed.
_STAGING_RE = re.compile(r"__stg_[0-9a-f]{8}$")


def staging_name(logical: str) -> str:
    """A unique, Qdrant-legal physical name for a staging build of ``logical``."""
    suffix = uuid.uuid4().hex[:8]
    base = f"{logical}__stg_{suffix}"
    return base[:63]


def _is_staging(name: str) -> bool:
    return bool(_STAGING_RE.search(name))


def _alias_map(client: QdrantClient) -> dict[str, str]:
    """{alias_name: physical_collection_name}."""
    try:
        return {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    except Exception:
        return {}


def _real_collections(client: QdrantClient) -> set[str]:
    try:
        return {c.name for c in client.get_collections().collections}
    except Exception:
        return set()


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = settings.qdrant_url
        if url == ":memory:" or url == "":
            _client = QdrantClient(":memory:", force_disable_check_same_thread=True)
        elif url.startswith("http"):
            _client = QdrantClient(url=url)
        else:
            _client = QdrantClient(path=url, force_disable_check_same_thread=True)
    return _client


def list_collections(client: QdrantClient) -> list[str]:
    """All collections, newest-created first (unknown ones alphabetically last).

    Qdrant does not expose creation timestamps, so we keep a creation-order
    registry in ``data/collection_order.json``.
    """
    physical = _real_collections(client)
    aliases = _alias_map(client)
    targeted = set(aliases.values())
    names: set[str] = set(aliases.keys())
    for p in physical:
        if p in targeted or _is_staging(p):
            continue
        names.add(p)
    idx = order_index()
    known = [n for n in names if n in idx]
    unknown = sorted(n for n in names if n not in idx)
    known.sort(key=lambda n: idx[n], reverse=True)
    return known + unknown


def delete_collection(client: QdrantClient, name: str) -> None:
    """Delete a logical collection, following an alias when one is in place."""
    from qdrant_client.models import DeleteAlias, DeleteAliasOperation

    with _write_lock:
        aliases = _alias_map(client)
        if name in aliases:
            target = aliases[name]
            client.update_collection_aliases(
                [DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=name))]
            )
            if target in _real_collections(client):
                client.delete_collection(target)
        elif name in _real_collections(client):
            client.delete_collection(name)
        drop_creation(name)


def set_active_collection(client: QdrantClient, name: str) -> None:
    """Switch the active collection at runtime (in-memory, resets on restart).

    Non-ASCII display names are mapped to a Qdrant-legal storage name; the
    display name persists via the name alias file.
    """
    if not name or not name.strip():
        raise ValueError("Collection name must not be empty")
    with _write_lock:
        storage = storage_name(name)
        register_alias(name, storage)
        record_creation(storage)
        settings.qdrant_collection = storage
        active.save(storage)
        ensure_collection(client, recreate=False)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    with _write_lock:
        ensure_collection_unlocked(client, recreate)


def ensure_collection_unlocked(client: QdrantClient, recreate: bool = False) -> None:
    name = current_collection()
    if client.collection_exists(name):
        if recreate:
            client.delete_collection(name)
        else:
            record_creation(name)
            return

    record_creation(name)
    create_collection(client, name)


def create_collection(client: QdrantClient, name: str) -> None:
    """Create an empty collection under an explicit physical name."""
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=embedding_dim(), distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": {},
        },
    )


def add_documents(
    client: QdrantClient,
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict] | None = None,
    collection: str | None = None,
) -> None:
    from qdrant_client.models import PointStruct, SparseVector

    points = []
    for i in range(len(texts)):
        vector = {"dense": dense_vectors[i]}
        if sparse_vectors and i < len(sparse_vectors):
            sv = sparse_vectors[i]
            vector["sparse"] = SparseVector(
                indices=sv["indices"],
                values=sv["values"],
            )

        points.append(
            PointStruct(
                id=ids[i],
                vector=vector,
                payload={
                    "text": texts[i],
                    "metadata": metadatas[i],
                },
            )
        )

    with _write_lock:
        client.upsert(
            collection_name=collection or current_collection(),
            points=points,
        )


def copy_source_points(
    client: QdrantClient, src: str, dst: str, source: str
) -> int:
    """Copy every point of ``source`` from collection ``src`` to ``dst``.

    Used to preserve unchanged files when an import rebuilds the whole
    collection in a staging collection. Batched so memory stays bounded.
    """
    from qdrant_client.models import PointStruct

    if not client.collection_exists(src):
        return 0
    copied = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=src,
            scroll_filter=_source_filter(source),
            limit=1000,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        if points:
            with _write_lock:
                client.upsert(
                    collection_name=dst,
                    points=[
                        PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                        for p in points
                    ],
                )
            copied += len(points)
        if offset is None:
            break
    return copied


def promote_collection(client: QdrantClient, logical: str, staging: str) -> None:
    """Atomically make ``logical`` resolve to the freshly built ``staging``.

    When ``logical`` is already an alias, the alias is repointed in a single
    atomic ``update_collection_aliases`` call, then the previous physical
    collection is dropped. On first promotion of a real collection the alias is
    created first (still shadowed by the real collection, so reads keep working)
    and the real collection is deleted afterwards — no read gap.
    """
    from qdrant_client.models import (
        CreateAlias,
        CreateAliasOperation,
        DeleteAlias,
        DeleteAliasOperation,
    )

    with _write_lock:
        aliases = _alias_map(client)
        if logical in aliases:
            previous = aliases[logical]
            client.update_collection_aliases(
                [
                    DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=logical)),
                    CreateAliasOperation(
                        create_alias=CreateAlias(
                            collection_name=staging, alias_name=logical
                        )
                    ),
                ]
            )
            if previous != staging and previous in _real_collections(client):
                client.delete_collection(previous)
        else:
            client.update_collection_aliases(
                [
                    CreateAliasOperation(
                        create_alias=CreateAlias(
                            collection_name=staging, alias_name=logical
                        )
                    )
                ]
            )
            if logical in _real_collections(client):
                client.delete_collection(logical)
        record_creation(logical)


def cleanup_staging(client: QdrantClient) -> int:
    """Drop orphaned staging collections from interrupted imports.

    A staging collection that is still referenced by an alias is a completed
    build awaiting use and is kept.
    """
    targeted = set(_alias_map(client).values())
    removed = 0
    for name in _real_collections(client):
        if _is_staging(name) and name not in targeted:
            client.delete_collection(name)
            removed += 1
    return removed


def collection_info(client: QdrantClient, collection: str | None = None) -> dict | None:
    name = collection or current_collection()
    if not client.collection_exists(name):
        return None
    info = client.get_collection(name)
    return {
        "name": name,
        "points_count": info.points_count,
        "status": info.status,
    }


def distinct_filenames(client: QdrantClient, collection: str | None = None) -> list[dict]:
    """Return [{filename, chunks}] of distinct filenames actually indexed."""
    stats = source_stats(client, collection=collection)
    filename_ids: dict[str, int] = {}
    for item in stats:
        key = Path(item["source"]).name
        filename_ids[key] = filename_ids.get(key, 0) + item["chunks"]
    return [
        {"filename": f, "chunks": c}
        for f, c in sorted(filename_ids.items(), key=lambda x: -x[1])
    ]


def source_stats(client: QdrantClient, collection: str | None = None) -> list[dict]:
    """Return [{source, filename, chunks, rel_path}] grouped by metadata.source."""
    name = collection or current_collection()
    if not client.collection_exists(name):
        return []
    stats: dict[str, dict] = {}
    offset: dict | None = None
    while True:
        points, next_offset = client.scroll(
            collection_name=name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            meta = (p.payload or {}).get("metadata", {})
            source = meta.get("source") or meta.get("rel_path") or meta.get("filename")
            if not source:
                continue
            entry = stats.setdefault(
                source,
                {
                    "source": source,
                    "filename": meta.get("filename", Path(source).name),
                    "chunks": 0,
                    "rel_path": meta.get("rel_path"),
                },
            )
            entry["chunks"] += 1
        if next_offset is None:
            break
        offset = next_offset
    return sorted(stats.values(), key=lambda x: x["source"])


def _source_filter(source: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="metadata.source",
                match=MatchValue(value=source),
            )
        ]
    )


def delete_by_source(client: QdrantClient, source: str, collection: str | None = None) -> int:
    """Delete all points whose metadata.source == source. Returns deleted count."""
    name = collection or current_collection()
    if not client.collection_exists(name):
        return 0
    points, _ = client.scroll(
        collection_name=name,
        scroll_filter=_source_filter(source),
        limit=1000,
        with_vectors=True,
    )
    if not points:
        return 0
    with _write_lock:
        client.delete(
            collection_name=name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source),
                    )
                ]
            ),
        )
    return len(points)
