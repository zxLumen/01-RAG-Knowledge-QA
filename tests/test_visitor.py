from __future__ import annotations

import time

import pytest

from src.api import visitor


@pytest.fixture
def vtmp(tmp_path, monkeypatch):
    monkeypatch.setattr(visitor, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(visitor, "SAMPLES_DIR", tmp_path / "data" / "samples")
    monkeypatch.setattr(visitor, "VISITORS_DIR", tmp_path / "data" / "visitors")
    monkeypatch.setattr(visitor, "STATE_PATH", tmp_path / "visitors.json")
    monkeypatch.setattr(visitor, "_state", None)
    # Avoid touching a real qdrant during tests.
    monkeypatch.setattr(visitor, "_delete_collection", lambda _vid: None)
    yield visitor
    monkeypatch.setattr(visitor, "_state", None)


def test_new_id_and_validity(vtmp):
    vid = vtmp.new_visitor_id()
    assert len(vid) == 16
    assert vtmp.is_valid_id(vid)
    assert not vtmp.is_valid_id("short")
    assert not vtmp.is_valid_id(None)
    assert not vtmp.is_valid_id("../etc")


def test_collection_and_dir(vtmp):
    vid = "a1b2c3d4e5f60718"
    assert vtmp.collection_name(vid) == "visitor_a1b2c3d4e5f60718"
    assert vtmp.visitor_dir(vid).name == vid
    with pytest.raises(ValueError):
        vtmp.collection_name("bad")


def test_touch_and_usage(vtmp):
    vid = vtmp.new_visitor_id()
    vtmp.touch(vid)
    assert vtmp.visitor_usage(vid) == 0
    d = vtmp.visitor_dir(vid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "a.txt").write_text("hello", encoding="utf-8")
    assert vtmp.visitor_usage(vid) == 5


def test_cleanup_expired(vtmp, monkeypatch):
    old = vtmp.new_visitor_id()
    fresh = vtmp.new_visitor_id()
    vtmp.touch(old)
    vtmp.touch(fresh)
    # age the old visitor beyond TTL
    state = vtmp._load_state()
    state["visitors"][old] = time.time() - vtmp.VISITOR_TTL_SECONDS - 10
    vtmp._save_state()
    removed = []
    monkeypatch.setattr(vtmp, "remove_visitor", lambda vid: removed.append(vid))
    vtmp.cleanup_expired()
    assert old in removed
    assert fresh not in removed


def test_remove_visitor_clears_dir_and_state(vtmp):
    vid = vtmp.new_visitor_id()
    vtmp.touch(vid)
    d = vtmp.visitor_dir(vid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "x.md").write_text("x", encoding="utf-8")
    vtmp.remove_visitor(vid)
    assert not d.exists()
    assert vid not in vtmp._load_state()["visitors"]


def test_enforce_global_quota_evicts_oldest(vtmp, monkeypatch):
    a = vtmp.new_visitor_id()
    b = vtmp.new_visitor_id()
    vtmp.touch(a)
    vtmp.touch(b)
    state = vtmp._load_state()
    state["visitors"][a] = time.time() - 100  # a is older
    state["visitors"][b] = time.time()
    vtmp._save_state()

    order = []
    monkeypatch.setattr(vtmp, "remove_visitor", lambda vid: order.append(vid))
    # Over quota until the first eviction, then under quota.
    usage = {"n": 0}

    def fake_usage():
        usage["n"] += 1
        # over quota for the initial check and the loop's first check (call #2)
        return vtmp.GLOBAL_QUOTA_BYTES + 1 if usage["n"] <= 2 else 0

    monkeypatch.setattr(vtmp, "global_usage", fake_usage)
    vtmp.enforce_global_quota(0)
    assert order == [a]
