"""One-off migration: embedded Qdrant storage → Qdrant server (on_disk).

Run with the rag container stopped (the embedded storage is exclusively
locked while the app runs). Intended to run inside the rag image with the
``rag-data`` volume mounted, e.g.::

    docker run --rm --network rag-internal \
        -v rag_rag-data:/app/qdrant_data \
        ghcr.io/zxlumen/01-rag-knowledge-qa:<tag> \
        python -m src.migrate_server --src /app/qdrant_data --dst http://qdrant:6333

It copies every collection (points + payload + vectors) preserving names and
recreates aliases, so the alias-based atomic swap keeps working. The source is
never modified; re-running is safe (existing target collections are skipped).
"""

from __future__ import annotations

import argparse
import sys

from qdrant_client import QdrantClient
from qdrant_client import models as qm


def _remote_vectors_config(local_info) -> dict:
    vectors = {}
    for name, params in (local_info.config.params.vectors or {}).items():
        vectors[name] = qm.VectorParams(
            size=params.size,
            distance=params.distance,
            on_disk=True,
        )
    sparse = {}
    for name, params in (local_info.config.params.sparse_vectors or {}).items():
        sparse[name] = qm.SparseVectorParams(
            index=qm.SparseIndexParams(on_disk=True)
        )
    return vectors, sparse


def migrate(src: str, dst: str, api_key: str | None, batch: int, dry_run: bool) -> int:
    local = QdrantClient(path=src, force_disable_check_same_thread=True)
    remote = QdrantClient(url=dst, api_key=api_key or None)

    aliases = {a.alias_name: a.collection_name for a in local.get_aliases().aliases}
    physical = [c.name for c in local.get_collections().collections]
    print(f"source collections: {physical}")
    print(f"source aliases: {aliases}")

    total = 0
    for name in physical:
        info = local.get_collection(name)
        count = info.points_count or 0
        if dry_run:
            print(f"[{name}] points={count} (dry-run)")
            total += count
            continue
        exists = remote.collection_exists(name)
        print(f"[{name}] points={count} target_exists={exists}")
        if not exists:
            vectors, sparse = _remote_vectors_config(info)
            remote.create_collection(
                collection_name=name,
                vectors_config=vectors,
                sparse_vectors_config=sparse,
                on_disk_payload=True,
            )
        offset = None
        while True:
            points, offset = local.scroll(
                collection_name=name,
                limit=batch,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            if points:
                remote.upsert(
                    collection_name=name,
                    points=[
                        qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                        for p in points
                    ],
                )
                total += len(points)
            if offset is None:
                break

    if not dry_run and aliases:
        ops = []
        for alias, target in aliases.items():
            if not remote.collection_exists(target):
                print(f"!! alias {alias} -> {target} skipped (target missing)")
                continue
            ops.append(
                qm.CreateAliasOperation(
                    create_alias=qm.CreateAlias(
                        collection_name=target, alias_name=alias
                    )
                )
            )
        if ops:
            remote.update_collection_aliases(ops)
            print(f"aliases recreated: {list(aliases)}")

    print(f"copied {total} points" + (" (dry-run)" if dry_run else ""))

    # verify
    if not dry_run:
        ok = True
        for name in physical:
            expected = local.get_collection(name).points_count or 0
            got = remote.get_collection(name).points_count or 0
            flag = "ok" if got == expected else "MISMATCH"
            if got != expected:
                ok = False
            print(f"verify [{name}] source={expected} target={got} {flag}")
        return 0 if ok else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="embedded storage path")
    parser.add_argument("--dst", required=True, help="Qdrant server URL")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return migrate(args.src, args.dst, args.api_key, args.batch, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
