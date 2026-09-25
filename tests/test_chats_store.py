
from src.chats import store


def _store(tmp_path, monkeypatch):
    monkeypatch.setattr(store.settings, "chat_db_path", str(tmp_path / "chat.db"))
    store.init_db()


def test_upsert_and_reactivate(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    row = store.upsert(
        "visitor_abc", "s1", title="会话 1", collection="visitor_abc",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert row["id"] == "s1"
    assert row["deleted"] is False
    assert row["messages"] == [{"role": "user", "content": "hi"}]

    # same id reactivates a soft-deleted row and keeps content when deleting
    store.upsert("visitor_abc", "s1", deleted=True)
    got = store.get_by_rowid(row["rowid"])
    assert got["deleted"] is True
    assert got["messages"] == [{"role": "user", "content": "hi"}]

    # a normal sync brings it back and updates content
    row = store.upsert(
        "visitor_abc", "s1", title="会话 1",
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
    )
    got = store.get_by_rowid(row["rowid"])
    assert got["deleted"] is False
    assert len(got["messages"]) == 2


def test_list_for_owner_filters_deleted(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    store.upsert("visitor_abc", "a", title="A", messages=[])
    store.upsert("visitor_abc", "b", title="B", messages=[])
    store.upsert("visitor_abc", "c", title="C", messages=[])
    row_b = store.get_by_rowid(store.upsert("visitor_abc", "b")["rowid"])
    store.set_deleted(row_b["rowid"], True)

    visible = [s["id"] for s in store.list_for_owner("visitor_abc")]
    assert visible == ["c", "a"]
    assert "b" not in visible


def test_admin_list_crosses_owners_and_hard_delete(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    r1 = store.upsert("visitor_abc", "a", title="A", messages=[])
    r2 = store.upsert(store.ADMIN_OWNER, "b", title="B", messages=[])
    rows = store.admin_list()
    assert {s["owner"] for s in rows} == {"visitor_abc", store.ADMIN_OWNER}
    assert store.hard_delete(r1["rowid"]) is True
    assert store.get_by_rowid(r1["rowid"]) is None
    assert store.get_by_rowid(r2["rowid"]) is not None


def test_schema_tolerates_missing_messages(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    row = store.upsert("visitor_xyz", "s9")
    assert row["messages"] == []
    assert row["deleted"] is False
