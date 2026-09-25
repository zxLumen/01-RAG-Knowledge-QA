from src.ingest import progress

OWNER = "admin"


def _reset():
    token = progress.bind(OWNER)
    progress.finish("done", 0, 0)
    progress.set_phase("", 0, 0)
    progress.unbind(token)


def test_progress_lifecycle():
    _reset()
    assert progress.begin(OWNER) is True
    snap = progress.snapshot(OWNER)
    assert snap["running"] is True
    assert snap["phase"] == "read"
    assert snap["started_at"] > 0
    assert snap["summary"] is None
    started_at = snap["started_at"]
    token = progress.bind(OWNER)
    try:
        progress.set_phase("embed", 0, 8)
        progress.update(4, 8)
        progress.finish("done", 1, 1, summary={"added": 4, "chunks": 8})
    finally:
        progress.unbind(token)
    snap = progress.snapshot(OWNER)
    assert snap["phase"] == "done"
    assert snap["running"] is False
    assert snap["started_at"] == started_at
    assert snap["summary"] == {"added": 4, "chunks": 8}
    assert progress.begin(OWNER) is True
    assert progress.snapshot(OWNER)["summary"] is None
    _reset()


def test_progress_snapshot_is_copy():
    _reset()
    progress.begin(OWNER)
    snap = progress.snapshot(OWNER)
    snap["running"] = False
    snap["done"] = 99
    assert progress.snapshot(OWNER)["running"] is True
    assert progress.snapshot(OWNER)["done"] == 0
    progress.finish("done", 0, 0)


def test_progress_cancel_flag_and_reset():
    progress.begin(OWNER)
    token = progress.bind(OWNER)
    try:
        assert progress.is_cancelled() is False
        progress.cancel(OWNER)
        assert progress.is_cancelled() is True
        progress.finish("cancelled", 0, 0)
    finally:
        progress.unbind(token)
    # a fresh run starts clean
    assert progress.begin(OWNER) is True
    token = progress.bind(OWNER)
    try:
        assert progress.is_cancelled() is False
        progress.finish("done", 0, 0)
    finally:
        progress.unbind(token)


def test_progress_cancel_run_scoping():
    progress.begin(OWNER)
    snap = progress.snapshot(OWNER)
    current = snap["started_at"]
    assert progress.is_current_run(None, snap) is True
    assert progress.is_current_run(current, snap) is True
    assert progress.is_current_run(current + 1, snap) is False
    assert progress.is_current_run(current - 60, snap) is False
    progress.finish("done", 1, 1)


def test_progress_is_isolated_per_owner():
    _reset()
    assert progress.begin("v:alice") is True
    # another owner sees nothing and cannot cancel
    assert progress.snapshot("v:bob")["running"] is False
    assert progress.cancel("v:bob") is False
    assert progress.snapshot("v:alice")["running"] is True
    # only one import runs globally
    assert progress.begin("v:bob") is False
    assert progress.cancel("v:alice") is True
    token = progress.bind("v:alice")
    assert progress.is_cancelled() is True
    progress.finish("cancelled", 0, 0)
    progress.unbind(token)
