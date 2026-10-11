from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

from src.retrieval import hybrid
from src.vectorstore import store


class _FakeDense:
    def __init__(self, vec):
        self.vec = vec

    def embed_query(self, text):
        return self.vec


class _FakeSparse:
    def __init__(self, indices, values):
        self._indices = indices
        self._values = values

    def embed_query(self, text):
        return SimpleNamespace(indices=self._indices, values=self._values)


@pytest.fixture
def client(monkeypatch):
    c = QdrantClient(":memory:")
    monkeypatch.setattr(store, "_client", c)
    monkeypatch.setattr(store, "embedding_dim", lambda: 4)
    yield c


def _add(client, pid, text, vec, sparse):
    store.add_documents(
        client,
        [text],
        [{"source": pid, "filename": f"{pid}.md"}],
        [str(uuid.uuid5(uuid.NAMESPACE_DNS, pid))],
        [vec],
        [sparse],
        collection="kb",
    )


def test_hybrid_sparse_promotes_lexical_match(client, monkeypatch):
    store.create_collection(client, "kb")
    # "alpha" is dense-best but has no sparse match; "beta" matches the sparse
    # query, so RRF should rank beta first.
    _add(client, "a", "alpha", [1.0, 0.0, 0.0, 0.0], {"indices": [1], "values": [1.0]})
    _add(client, "b", "beta", [0.0, 1.0, 0.0, 0.0], {"indices": [9], "values": [1.0]})

    monkeypatch.setattr(hybrid, "get_dense_embeddings", lambda: _FakeDense([1.0, 0.0, 0.0, 0.0]))
    monkeypatch.setattr(hybrid, "get_sparse_embeddings", lambda: _FakeSparse([9], [1.0]))
    monkeypatch.setattr(hybrid, "_collection_has_sparse", lambda c, n: True)
    monkeypatch.setattr(hybrid.retrieval_config, "load", lambda: {
        **hybrid.retrieval_config._defaults(),
        "hybrid_enabled": True,
        "keyword_limit": 0,
        "top_k": 5,
    })

    results = hybrid.search("beta", collection_name="kb")
    texts = [r.text for r in results]
    assert texts[0] == "beta"
    assert set(texts) == {"alpha", "beta"}


def test_dense_only_fallback_when_no_sparse(client, monkeypatch):
    store.create_collection(client, "kb")
    _add(client, "a", "alpha", [1.0, 0.0, 0.0, 0.0], {"indices": [1], "values": [1.0]})

    monkeypatch.setattr(hybrid, "get_dense_embeddings", lambda: _FakeDense([1.0, 0.0, 0.0, 0.0]))
    monkeypatch.setattr(hybrid, "_collection_has_sparse", lambda c, n: False)
    monkeypatch.setattr(hybrid.retrieval_config, "load", lambda: {
        **hybrid.retrieval_config._defaults(),
        "hybrid_enabled": True,
        "keyword_limit": 0,
        "top_k": 5,
    })

    results = hybrid.search("alpha", collection_name="kb")
    assert [r.text for r in results] == ["alpha"]


def test_missing_collection_returns_empty(client, monkeypatch):
    monkeypatch.setattr(hybrid, "get_dense_embeddings", lambda: _FakeDense([1.0, 0.0, 0.0, 0.0]))
    assert hybrid.search("anything", collection_name="does_not_exist") == []
