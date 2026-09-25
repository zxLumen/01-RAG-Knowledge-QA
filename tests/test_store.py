import uuid

from qdrant_client import QdrantClient

from src.config import settings
from src.vectorstore.store import (
    add_documents,
    delete_by_source,
    distinct_filenames,
    ensure_collection,
)


def test_distinct_filenames_groups_by_metadata(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_collection", "test_kb")
    client = QdrantClient(":memory:")
    ensure_collection(client)

    dim = settings.embedding_dim
    add_documents(
        client,
        texts=["a", "b", "c"],
        metadatas=[
            {"filename": "x.md"},
            {"filename": "x.md"},
            {"filename": "y.md"},
        ],
        ids=[str(uuid.uuid4()) for _ in range(3)],
        dense_vectors=[[0.1] * dim, [0.2] * dim, [0.3] * dim],
    )

    result = distinct_filenames(client)
    by_name = {item["filename"]: item["chunks"] for item in result}
    assert by_name == {"x.md": 2, "y.md": 1}


def test_distinct_filenames_empty_collection(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_collection", "test_empty")
    client = QdrantClient(":memory:")
    assert distinct_filenames(client) == []


def test_delete_by_source_scoped_to_collection(monkeypatch):
    dim = settings.embedding_dim
    client = QdrantClient(":memory:")

    monkeypatch.setattr(settings, "qdrant_collection", "kb_a")
    ensure_collection(client)
    add_documents(
        client,
        texts=["a1", "a2"],
        metadatas=[{"source": "x.md"}, {"source": "x.md"}],
        ids=[str(uuid.uuid4()) for _ in range(2)],
        dense_vectors=[[0.1] * dim, [0.2] * dim],
    )

    monkeypatch.setattr(settings, "qdrant_collection", "kb_b")
    ensure_collection(client)
    add_documents(
        client,
        texts=["b1"],
        metadatas=[{"source": "x.md"}],
        ids=[str(uuid.uuid4())],
        dense_vectors=[[0.3] * dim],
    )

    assert delete_by_source(client, "x.md", collection="kb_a") == 2
    assert distinct_filenames(client, collection="kb_a") == []
    by_b = {i["filename"]: i["chunks"] for i in distinct_filenames(client, collection="kb_b")}
    assert by_b == {"x.md": 1}
