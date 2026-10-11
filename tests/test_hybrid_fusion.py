from __future__ import annotations

from src.retrieval.hybrid import (
    _candidate_pool_size,
    _extract_keywords,
    _rrf_fuse,
    _select_final_chunks,
)


def _item(point_id: object, score: float = 0.0, source: str = "a") -> dict:
    return {"id": point_id, "text": "", "score": score, "source": source, "metadata": {}}


def test_rrf_prefers_items_present_in_both_channels():
    dense = [_item(1), _item(2), _item(3)]
    sparse = [_item(2), _item(4)]
    fused = _rrf_fuse([dense, sparse], 60, [1.0, 1.0])
    assert fused[0]["id"] == 2


def test_rrf_respects_channel_weights():
    dense = [_item(1), _item(2)]
    sparse = [_item(2), _item(1)]
    fused = _rrf_fuse([dense, sparse], 60, [0.0, 1.0])
    assert fused[0]["id"] == 2
    assert fused[1]["id"] == 1


def test_select_final_chunks_limits_files_and_chunks_per_file():
    items = [_item(i, score=10 - i, source=f"f{i % 2}") for i in range(6)]
    out = _select_final_chunks(items, top_k=1, chunks_per_file=2)
    assert len({x["source"] for x in out}) == 1
    assert len(out) <= 2


def test_select_final_chunks_keeps_multiple_files_up_to_top_k():
    items = [_item(i, score=10 - i, source=f"f{i}") for i in range(5)]
    out = _select_final_chunks(items, top_k=3, chunks_per_file=1)
    assert len({x["source"] for x in out}) == 3


def test_candidate_pool_uses_min_then_multiplier():
    cfg = {"candidate_multiplier": 4, "candidate_min": 24}
    assert _candidate_pool_size(3, cfg) == 24
    assert _candidate_pool_size(20, cfg) == 80


def test_extract_keywords_keeps_alnum_and_cjk():
    kws = _extract_keywords("郭建美的身份证号 370627197110280241")
    assert "370627197110280241" in kws
    assert "郭建美的身份证号" in kws
