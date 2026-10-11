from __future__ import annotations

import pytest

from src.api import visitor


@pytest.fixture(autouse=True)
def _isolate_mint_state(tmp_path, monkeypatch):
    """Keep visitor mint-rate-limit state out of the real qdrant_data.

    The counter is intentionally persisted to disk in production; tests that
    mint visitors would otherwise accumulate hits across runs and start
    returning 429 once the per-IP hourly cap is reached.
    """
    monkeypatch.setattr(visitor, "MINT_STATE_PATH", tmp_path / "mint_hits.json")
