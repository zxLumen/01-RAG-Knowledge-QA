from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingest.loader import LoadedDocument
from src.retrieval import config as retrieval_config


def split_documents(
    documents: list[LoadedDocument],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    default_size, default_overlap = retrieval_config.chunk_settings()
    chunk_size = chunk_size or default_size
    chunk_overlap = chunk_overlap or default_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", ".", " ", ""],
        length_function=len,
    )

    chunks: list[dict] = []
    for doc in documents:
        texts = splitter.split_text(doc.content)
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        **doc.metadata,
                        "chunk_index": i,
                    },
                }
            )
    return chunks
