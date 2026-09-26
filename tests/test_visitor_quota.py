import time
import uuid

import src.config as config
import src.vectorstore.store as store
from src.api import visitor


def _fake_embeddings():
    dim = config.settings.embedding_dim

    class FakeDense:
        def embed_documents(self, texts):
            return [[0.0] * dim for _ in texts]

        def embed_query(self, text):
            return [0.0] * dim

    def _dense():
        return FakeDense()

    def _sparse():
        raise RuntimeError("sparse disabled in tests")

    return _dense, _sparse


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.settings.import_db_path", str(tmp_path / "imports.db"))
    monkeypatch.setattr("src.config.settings.qdrant_url", ":memory:")
    monkeypatch.setattr("src.config.settings.qdrant_collection", "test_kb")
    monkeypatch.setattr(visitor, "STATE_PATH", tmp_path / "visitors.json")
    monkeypatch.setattr(visitor, "VISITORS_DIR", tmp_path / "visitors")
    visitor._state = None
    import src.api.routes as routes

    monkeypatch.setattr(routes, "_mint_hits", {})
    store._client = None


def _add_points(collection, n, monkeypatch):
    from src.vectorstore.store import add_documents, create_collection

    dense, _ = _fake_embeddings()
    client = store.get_client()
    create_collection(client, collection)
    add_documents(
        client,
        texts=[f"t{i}" for i in range(n)],
        metadatas=[{"source": f"{collection}.md"} for _ in range(n)],
        ids=[str(uuid.uuid4()) for _ in range(n)],
        dense_vectors=[[0.0] * config.settings.embedding_dim for _ in range(n)],
        collection=collection,
    )


def test_ingest_point_limit_rejects_and_keeps_collection(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from src.ingest import pipeline
    from src.vectorstore.store import collection_points, list_collections

    dense, sparse = _fake_embeddings()
    monkeypatch.setattr(pipeline, "get_dense_embeddings", dense)
    monkeypatch.setattr(pipeline, "get_sparse_embeddings", sparse)

    d = tmp_path / "kb"
    d.mkdir()
    f = d / "a.md"
    f.write_text("hello world. " * 200, encoding="utf-8")

    r = pipeline.ingest_paths([str(d)], recreate=True, point_limit=50)
    assert r["chunks"] >= 3
    before = collection_points(store.get_client(), config.settings.qdrant_collection)
    assert before == r["chunks"]

    # change the file so it is re-embedded, then reject with a tiny limit
    f.write_text("changed content. " * 200, encoding="utf-8")
    r2 = pipeline.ingest_paths([str(d)], recreate=False, point_limit=2)
    assert "error" in r2
    assert "上限" in r2["error"]
    client = store.get_client()
    assert collection_points(client, config.settings.qdrant_collection) == before
    assert not any("__stg_" in n for n in list_collections(client))


def test_global_eviction_only_touches_visitor_collections(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from src.vectorstore.store import collection_points

    # admin collection must survive every path
    _add_points("knowledge_base", 3, monkeypatch)

    old_vid = "a" * 16
    new_vid = "b" * 16
    _add_points(visitor.collection_name(old_vid), 5, monkeypatch)
    _add_points(visitor.collection_name(new_vid), 5, monkeypatch)

    # an upload from the old visitor must survive eviction
    up = visitor.visitor_dir(old_vid)
    up.mkdir(parents=True, exist_ok=True)
    (up / "keep.txt").write_text("keep me", encoding="utf-8")

    state = visitor._load_state()
    state["visitors"][old_vid] = time.time() - 1000
    state["visitors"][new_vid] = time.time()
    visitor._state = state

    client = store.get_client()
    evicted = visitor.ensure_global_capacity(new_vid, need_points=5, budget=8, client=client)

    assert evicted == [old_vid]
    assert not client.collection_exists(visitor.collection_name(old_vid))
    # current visitor and admin untouched
    assert client.collection_exists(visitor.collection_name(new_vid))
    assert client.collection_exists("knowledge_base")
    assert collection_points(client, "knowledge_base") == 3
    # upload preserved
    assert (up / "keep.txt").exists()

    _cleanup(client, old_vid, new_vid)


def test_global_capacity_raises_when_nothing_to_evict(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from src.ingest.pipeline import QuotaExceededError

    vid = "c" * 16
    state = visitor._load_state()
    state["visitors"][vid] = time.time()
    visitor._state = state

    try:
        visitor.ensure_global_capacity(vid, need_points=10, budget=5, client=store.get_client())
    except QuotaExceededError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected QuotaExceededError")


def test_mint_rate_limit(tmp_path, monkeypatch):
    import src.api.routes as routes

    monkeypatch.setattr(routes.settings, "visitor_mint_per_hour", 2)
    monkeypatch.setattr(routes, "_mint_hits", {})

    assert routes._allow_new_identity("1.2.3.4") is True
    assert routes._allow_new_identity("1.2.3.4") is True
    assert routes._allow_new_identity("1.2.3.4") is False
    # other IPs unaffected
    assert routes._allow_new_identity("5.6.7.8") is True

    # 0 disables the limit
    monkeypatch.setattr(routes.settings, "visitor_mint_per_hour", 0)
    assert routes._allow_new_identity("1.2.3.4") is True


def _cleanup(client, *vids):
    from src.vectorstore.store import delete_collection

    for vid in vids:
        name = visitor.collection_name(vid)
        if client.collection_exists(name):
            delete_collection(client, name)
