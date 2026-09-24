from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import requests as _requests

from src.config import settings
from src.ingest import progress as ingest_progress
from src.qa import providers
from src.qa.llm_config import get_embedding, resolve_key

EMBED_TIMEOUT = 300
EMBED_BATCH_SIZE = 96
EMBED_WORKERS = 1


def _embed_headers(api_key: str) -> dict:
    key = resolve_key(api_key)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _post_embeddings(endpoint: str, payload: dict, api_key: str) -> dict:
    import time as _time

    last: Exception | None = None
    for attempt in range(3):
        try:
            resp = _requests.post(
                endpoint,
                json=payload,
                headers=_embed_headers(api_key),
                timeout=EMBED_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except (_requests.exceptions.ConnectionError, _requests.exceptions.Timeout) as e:
            last = e
            if attempt < 2:
                _time.sleep(2.0 * (attempt + 1))
    raise last  # type: ignore[misc]


@dataclass
class SparseEmbeddingResult:
    indices: list[int]
    values: list[float]


class DenseEmbeddings:
    """Dense embeddings via the configured provider (Ollama or OpenAI-compatible)."""

    def __init__(self, model_name: str | None = None):
        cfg = get_embedding()
        self.protocol = cfg.get("protocol") or providers.protocol_for(cfg.get("provider"))
        self.base_url = (cfg.get("base_url") or "").rstrip("/")
        self.api_key = cfg.get("api_key", "")
        self.model_name = model_name or cfg.get("model") or settings.dense_embedding_model

    def _endpoint(self) -> str:
        if self.protocol == providers.PROTOCOL_OLLAMA:
            return f"{self.base_url}/api/embed"
        return f"{self.base_url}/embeddings"

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if self.protocol == providers.PROTOCOL_OLLAMA:
            payload = {"model": self.model_name, "input": batch, "keep_alive": "30m"}
            result = _post_embeddings(self._endpoint(), payload, self.api_key)
            return result["embeddings"]
        payload = {"model": self.model_name, "input": batch}
        result = _post_embeddings(self._endpoint(), payload, self.api_key)
        return [item["embedding"] for item in result["data"]]

    def embed_documents(
        self, texts: list[str], progress: Callable[[int], None] | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        batches = [
            texts[i : i + EMBED_BATCH_SIZE]
            for i in range(0, len(texts), EMBED_BATCH_SIZE)
        ]

        def run(batch: list[str]) -> list[list[float]]:
            return self._embed_batch(batch)

        if len(batches) == 1:
            if ingest_progress.is_cancelled():
                raise RuntimeError("cancelled")
            result = run(batches[0])
            if progress:
                progress(len(texts))
            return result
        results: dict[int, list[list[float]]] = {}
        done_count = 0
        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
            futures = {
                executor.submit(run, batch): (i, len(batch))
                for i, batch in enumerate(batches)
            }
            for fut in as_completed(futures):
                if ingest_progress.is_cancelled():
                    raise RuntimeError("cancelled")
                i, size = futures[fut]
                results[i] = fut.result()
                done_count += size
                if progress:
                    progress(done_count)
        return [vector for i in range(len(batches)) for vector in results[i]]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


class LocalSparseEmbeddings:
    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(
        self, texts: list[str], progress: Callable[[int], None] | None = None
    ) -> list[SparseEmbeddingResult]:
        model = self._get_model()
        results = []
        for s in model.embed(texts):
            if ingest_progress.is_cancelled():
                raise RuntimeError("cancelled")
            results.append(
                SparseEmbeddingResult(indices=s.indices.tolist(), values=s.values.tolist())
            )
            if progress:
                progress(1)
        return results

    def embed_query(self, text: str) -> SparseEmbeddingResult:
        model = self._get_model()
        results = list(model.embed([text]))
        result = results[0]
        return SparseEmbeddingResult(
            indices=result.indices.tolist(), values=result.values.tolist()
        )


def get_dense_embeddings() -> DenseEmbeddings:
    return DenseEmbeddings()


def get_sparse_embeddings() -> LocalSparseEmbeddings:
    return LocalSparseEmbeddings()
