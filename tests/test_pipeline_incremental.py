def _fake_embeddings():
    class FakeDense:
        def __init__(self):
            import src.config as config
            self.dim = config.settings.embedding_dim

        def embed_documents(self, texts):
            return [[0.0] * self.dim for _ in texts]

        def embed_query(self, text):
            return [0.0] * self.dim

    def _dense():
        return FakeDense()

    def _sparse():
        raise RuntimeError("sparse disabled in tests")

    return _dense, _sparse


def _setup_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.settings.import_db_path", str(tmp_path / "imports.db"))
    monkeypatch.setattr("src.config.settings.qdrant_url", ":memory:")
    monkeypatch.setattr("src.config.settings.qdrant_collection", "test_inc")
    import src.vectorstore.store as store
    store._client = None


def _count_points():
    from src.config import settings
    from src.vectorstore.store import get_client
    return get_client().count(collection_name=settings.qdrant_collection).count


def test_incremental_add_update_unchanged_delete(tmp_path, monkeypatch):
    _setup_paths(tmp_path, monkeypatch)
    dense, sparse = _fake_embeddings()
    monkeypatch.setattr("src.ingest.pipeline.get_dense_embeddings", dense)
    monkeypatch.setattr("src.ingest.pipeline.get_sparse_embeddings", sparse)

    from src.ingest.pipeline import ingest_paths

    data_dir = tmp_path / "kb"
    data_dir.mkdir()
    f1 = data_dir / "a.md"
    f2 = data_dir / "b.md"
    f1.write_text("hello world content alpha. " * 20, encoding="utf-8")
    f2.write_text("second file content beta. " * 20, encoding="utf-8")
    root = str(data_dir)

    r = ingest_paths([root], recreate=True, delete_missing=True)
    assert r["added"] == 2 and r["chunks"] == 2
    assert _count_points() == 2

    r = ingest_paths([root], recreate=False, delete_missing=True)
    assert r["added"] == 0 and r["updated"] == 0
    assert r["unchanged"] == 2 and r["deleted"] == 0 and r["chunks"] == 0
    assert _count_points() == 2

    f1.write_text("hello world content alpha CHANGED. " * 20, encoding="utf-8")
    r = ingest_paths([root], recreate=False, delete_missing=True)
    assert r["updated"] == 1 and r["unchanged"] == 1
    assert r["chunks"] == 1
    assert _count_points() == 2

    f1.unlink()
    r = ingest_paths([root], recreate=False, delete_missing=True)
    assert r["deleted"] == 1 and r["unchanged"] == 1 and r["chunks"] == 0
    assert _count_points() == 1

    from src.imports.store import latest_md5_by_rel_path
    assert latest_md5_by_rel_path().get(str(f1), {}).get("file_md5") is None


def test_batch_dedupes_paths(tmp_path, monkeypatch):
    _setup_paths(tmp_path, monkeypatch)
    dense, sparse = _fake_embeddings()
    monkeypatch.setattr("src.ingest.pipeline.get_dense_embeddings", dense)
    monkeypatch.setattr("src.ingest.pipeline.get_sparse_embeddings", sparse)

    from src.ingest.pipeline import ingest_paths

    data_dir = tmp_path / "kb"
    data_dir.mkdir()
    (data_dir / "a.md").write_text("alpha beta gamma. " * 20, encoding="utf-8")
    root = str(data_dir)

    r = ingest_paths([root, root, str(data_dir / "a.md")], recreate=True)
    assert r["added"] == 1 and r["documents"] == 1

    r = ingest_paths([str(data_dir / "a.md")], recreate=False, delete_missing=True)
    assert r["unchanged"] == 1


def test_delete_missing_respects_scope(tmp_path, monkeypatch):
    _setup_paths(tmp_path, monkeypatch)
    dense, sparse = _fake_embeddings()
    monkeypatch.setattr("src.ingest.pipeline.get_dense_embeddings", dense)
    monkeypatch.setattr("src.ingest.pipeline.get_sparse_embeddings", sparse)

    from src.ingest.pipeline import ingest_paths

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "x.md").write_text("something in a. " * 20, encoding="utf-8")
    (dir_b / "y.md").write_text("something in b. " * 20, encoding="utf-8")

    ingest_paths([str(tmp_path)], recreate=True)

    (dir_a / "x.md").unlink()
    r = ingest_paths([str(dir_a)], recreate=False, delete_missing=True)
    assert r["deleted"] == 1

    # scope check: deleting from dir_a must not touch dir_b's points
    assert _count_points() == 1

    from src.imports.store import latest_md5_by_rel_path
    assert latest_md5_by_rel_path().get(str(dir_a / "x.md"), {}).get("file_md5") is None


def test_collection_isolation_and_switch(tmp_path, monkeypatch):
    _setup_paths(tmp_path, monkeypatch)
    dense, sparse = _fake_embeddings()
    monkeypatch.setattr("src.ingest.pipeline.get_dense_embeddings", dense)
    monkeypatch.setattr("src.ingest.pipeline.get_sparse_embeddings", sparse)

    from src.ingest.pipeline import ingest_paths
    from src.vectorstore.store import get_client, list_collections, set_active_collection

    data_dir = tmp_path / "kb"
    data_dir.mkdir()
    f = data_dir / "a.md"
    f.write_text("alpha beta gamma. " * 20, encoding="utf-8")

    client = get_client()
    set_active_collection(client, "coll_a")
    r = ingest_paths([str(data_dir)], recreate=True)
    assert r["added"] == 1
    assert _count_points() == 1

    set_active_collection(client, "coll_b")
    assert _count_points() == 0
    r = ingest_paths([str(data_dir)], recreate=True)
    assert r["added"] == 1

    assert set(list_collections(client)) == {"coll_a", "coll_b"}

    r = ingest_paths([str(data_dir)], recreate=False, delete_missing=True)
    assert r["added"] == 0 and r["unchanged"] == 1

    set_active_collection(client, "coll_a")
    r = ingest_paths([str(data_dir)], recreate=False, delete_missing=True)
    assert r["unchanged"] == 1
    assert _count_points() == 1


def _point_texts():
    from src.config import settings
    from src.vectorstore.store import get_client

    points, _ = get_client().scroll(
        collection_name=settings.qdrant_collection, limit=100, with_payload=True
    )
    return sorted(p.payload["text"] for p in points)


def test_failed_import_leaves_collection_unchanged(tmp_path, monkeypatch):
    _setup_paths(tmp_path, monkeypatch)
    dense, sparse = _fake_embeddings()
    monkeypatch.setattr("src.ingest.pipeline.get_dense_embeddings", dense)
    monkeypatch.setattr("src.ingest.pipeline.get_sparse_embeddings", sparse)

    from src.ingest.pipeline import ingest_paths
    from src.vectorstore.store import get_client, list_collections

    data_dir = tmp_path / "kb"
    data_dir.mkdir()
    (data_dir / "a.md").write_text("alpha content. " * 20, encoding="utf-8")
    (data_dir / "b.md").write_text("beta content. " * 20, encoding="utf-8")

    ingest_paths([str(data_dir)], recreate=True)
    before = _point_texts()
    assert len(before) == 2

    # second import fails after the first embed window
    import src.ingest.pipeline as pipeline

    monkeypatch.setattr(pipeline, "INGEST_BATCH_SIZE", 1)

    class FailingDense:
        calls = 0
        dim = 1024

        def embed_documents(self, texts):
            FailingDense.calls += 1
            if FailingDense.calls >= 2:
                raise RuntimeError("boom")
            return [[0.0] * self.dim for _ in texts]

    monkeypatch.setattr(pipeline, "get_dense_embeddings", lambda: FailingDense())

    (data_dir / "a.md").write_text("alpha CHANGED. " * 20, encoding="utf-8")
    (data_dir / "b.md").write_text("beta CHANGED. " * 20, encoding="utf-8")
    r = ingest_paths([str(data_dir)], recreate=False, delete_missing=True)
    assert r.get("error")

    # collection unchanged and no staging leftovers
    assert _point_texts() == before
    assert not any("__stg_" in name for name in list_collections(get_client()))


def test_recreate_prunes_stale_ledger_for_other_roots(tmp_path, monkeypatch):
    _setup_paths(tmp_path, monkeypatch)
    dense, sparse = _fake_embeddings()
    monkeypatch.setattr("src.ingest.pipeline.get_dense_embeddings", dense)
    monkeypatch.setattr("src.ingest.pipeline.get_sparse_embeddings", sparse)

    from src.imports.store import latest_md5_by_rel_path
    from src.ingest.pipeline import ingest_paths

    kb = tmp_path / "kb"
    other = tmp_path / "other"
    kb.mkdir()
    other.mkdir()
    (kb / "a.md").write_text("alpha beta gamma. " * 20, encoding="utf-8")
    (other / "b.md").write_text("delta epsilon zeta. " * 20, encoding="utf-8")

    ingest_paths([str(kb), str(other)], recreate=True)
    assert str(other / "b.md") in latest_md5_by_rel_path()

    ingest_paths([str(kb)], recreate=True)
    assert str(other / "b.md") not in latest_md5_by_rel_path()
    assert str(kb / "a.md") in latest_md5_by_rel_path()

    r = ingest_paths([str(other / "b.md")], recreate=False, delete_missing=True)
    assert r["added"] == 1 and r["unchanged"] == 0
