#!/usr/bin/env python3
"""Retrieval-only evaluation (no LLM, no local Ollama).

Measures whether the expected source file appears among the retrieved chunks,
using the ``/api/retrieve`` endpoint. Reports recall@k over distinct files and
MRR (mean reciprocal rank of the first hit).

Usage::

    python tests/eval_retrieval.py                              # localhost:8000
    python tests/eval_retrieval.py --api https://rag.zxlumen.cn/api --top-k 5
    python tests/eval_retrieval.py --collection <name> --admin-token <token>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

TEST_FILE = Path(__file__).parent / "rag_eval.json"


def retrieve(
    session: requests.Session,
    api: str,
    question: str,
    top_k: int,
    collection: str | None,
    token: str | None,
) -> dict:
    body: dict = {"question": question, "top_k": top_k}
    if collection:
        body["collection"] = collection
    headers = {"X-Admin-Token": token} if token else None
    resp = session.post(f"{api}/retrieve", json=body, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def first_rank(sources: list[str], expected: str) -> int:
    for i, name in enumerate(sources):
        if expected and (expected in name or name in expected):
            return i
    return -1


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieval-only RAG evaluation")
    ap.add_argument("--api", default="http://localhost:8000/api")
    ap.add_argument("--top-k", type=int, default=5, help="distinct files to request")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--admin-token", default=None)
    ap.add_argument("--test-file", default=str(TEST_FILE))
    args = ap.parse_args()

    with open(args.test_file, encoding="utf-8") as f:
        cases = json.load(f)

    print(f"{len(cases)} cases | api={args.api} top_k={args.top_k}\n")

    # Reuse one session so the visitor cookie (demo mode) is minted once instead
    # of on every request, which would trip the per-IP mint rate limit.
    session = requests.Session()

    hits = 0
    mrr = 0.0
    total = 0
    by_cat: dict[str, dict] = {}

    for tc in cases:
        cat = tc.get("category", "?")
        st = by_cat.setdefault(cat, {"n": 0, "hits": 0, "mrr": 0.0})
        total += 1
        st["n"] += 1
        try:
            data = retrieve(
                session,
                args.api,
                tc["question"],
                args.top_k,
                args.collection,
                args.admin_token,
            )
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"[{tc['id']:2d}] ERROR {exc}")
            continue

        sources = data.get("sources", [])
        rank = first_rank(sources, tc.get("expected_source", ""))
        if rank >= 0:
            hits += 1
            st["hits"] += 1
            gain = 1.0 / (rank + 1)
            mrr += gain
            st["mrr"] += gain
            print(f"[{tc['id']:2d}] HIT  rank={rank + 1} {cat}")
        else:
            print(f"[{tc['id']:2d}] MISS {cat} | {tc['question']}")
            print(f"      expected={tc.get('expected_source')} got={sources}")

    print(f"\n{'=' * 60}")
    recall = hits / total * 100 if total else 0.0
    print(f"recall@{args.top_k} = {hits}/{total} ({recall:.1f}%) | MRR = {mrr / total:.3f}")

    print("\nBy category:")
    for cat, st in by_cat.items():
        pct = st["hits"] / st["n"] * 100 if st["n"] else 0.0
        cat_mrr = st["mrr"] / st["n"] if st["n"] else 0.0
        print(f"  {cat}: recall {st['hits']}/{st['n']} ({pct:.1f}%) | MRR {cat_mrr:.3f}")

    return 0 if hits == total else 1


if __name__ == "__main__":
    sys.exit(main())
