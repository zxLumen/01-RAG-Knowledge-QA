import threading

from src.imports import store


def test_store_threadsafe_across_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(store.settings, "import_db_path", str(tmp_path / "i.db"))
    store.init_db()
    results: list[tuple[int, str]] = []

    def worker(tag: str):
        sid = store.add_session(
            f"/tmp/{tag}", recreate=False, documents=1, chunks=1, status="ok"
        )
        store.add_files(sid, [{"filename": f"{tag}.md", "rel_path": f"/tmp/{tag}.md",
                               "status": "added", "chunk_count": 1, "file_size": 3,
                               "file_md5": "abc"}])
        results.append((sid, tag))

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 6
    rows = store.list_sessions(collection=store.settings.qdrant_collection)
    assert len(rows) == 6
    assert {r["path"] for r in rows} == {f"/tmp/t{i}" for i in range(6)}


def test_record_deletion_tombstone(tmp_path, monkeypatch):
    monkeypatch.setattr(store.settings, "import_db_path", str(tmp_path / "d.db"))
    store.init_db()
    coll = "kb_del"
    sid = store.add_session(
        "data", recreate=False, documents=1, chunks=2, status="ok", collection=coll
    )
    store.add_files(
        sid,
        [
            {
                "filename": "a.md",
                "rel_path": "data/a.md",
                "status": "added",
                "chunk_count": 2,
                "file_size": 5,
                "file_md5": "abc",
            }
        ],
    )
    assert store.latest_md5_by_rel_path(coll)["data/a.md"]["file_md5"] == "abc"

    store.record_deletion(coll, ["data/a.md"])

    # Tombstone clears the md5 so a later re-upload is re-indexed.
    assert store.latest_md5_by_rel_path(coll)["data/a.md"]["file_md5"] is None
    stats = store.collection_session_stats(coll)
    latest = stats[max(stats)]
    assert latest["deleted"] == 1
    assert latest["total_files"] == 0
    assert latest["total_chunks"] == 0
