from __future__ import annotations

import logging
from collections.abc import Generator

from src.config import settings
from src.qa import cancel, llm
from src.qa.llm_config import get_active
from src.retrieval.hybrid import RetrievedChunk, search

logger = logging.getLogger("rag")

SYSTEM_PROMPT = """你是一个知识库问答助手。基于以下检索到的文档片段回答用户问题。

规则:
1. 只基于提供的文档回答，不要编造信息
2. 文档中的具体数字、数值、编号、名称、列表项必须原样引用，不得省略、不得改写、不得概括
3. 如果文档中确实没有相关信息，才回答"根据现有文档，我无法回答这个问题"
4. 引用来源: 在回答中标注 [来源: 文件名]
5. 回答简洁准确，先直接给出答案"""


MAX_CONTEXT_CHARS = 4000


def _build_context(chunks: list[RetrievedChunk]) -> tuple[list[str], str]:
    context_parts = []
    sources = []
    budget = MAX_CONTEXT_CHARS
    for chunk in chunks:
        filename = chunk.metadata.get("filename", "unknown")
        page = chunk.metadata.get("page")
        loc = f"{filename}" + (f" (page {page})" if page else "")
        part = f"[{len(context_parts) + 1}] ({loc})\n{chunk.text}"
        if budget <= 0:
            break
        context_parts.append(part)
        sources.append(loc)
        budget -= len(part)
    return sources, "\n\n".join(context_parts)


def _make_messages(context: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"检索到的文档片段:\n\n{context}\n\n---\n用户问题: {question}",
        },
    ]


def _chunk_details(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "filename": c.metadata.get("filename", "unknown"),
            "text": c.text,
            "score": round(c.score, 4),
        }
        for c in chunks
    ]


def answer_question(
    question: str, top_k: int | None = None, collection_name: str | None = None
) -> dict:
    chunks = search(question, top_k=top_k or settings.top_k, collection_name=collection_name)
    if not chunks:
        return {
            "answer": "知识库中没有找到相关文档。",
            "sources": [],
            "chunks_used": 0,
            "retrieved": [],
        }

    sources, context = _build_context(chunks)

    profile = get_active()
    answer = llm.complete(profile, _make_messages(context, question))

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(sources),
        "retrieved": _chunk_details(chunks),
    }


def answer_question_stream(
    question: str,
    top_k: int | None = None,
    collection_name: str | None = None,
    gen_id: str | None = None,
) -> Generator[dict, None, None]:
    try:
        chunks = search(
            question, top_k=top_k or settings.top_k, collection_name=collection_name
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the client below
        logger.exception("retrieval failed (collection=%s)", collection_name)
        yield {
            "type": "done",
            "answer": f"检索失败：{exc}",
            "sources": [],
            "chunks_used": 0,
            "retrieved": [],
            "stopped": False,
            "error": True,
        }
        return
    if not chunks:
        yield {
            "type": "done",
            "answer": "知识库中没有找到相关文档。",
            "sources": [],
            "chunks_used": 0,
            "retrieved": [],
        }
        return

    sources, context = _build_context(chunks)
    stopped = False
    full_answer = ""
    profile = get_active()
    messages = _make_messages(context, question)
    cancel.register(gen_id)

    try:
        for text in llm.stream_chat(profile, messages, gen_id):
            full_answer += text
            yield {"type": "chunk", "text": text}
        stopped = cancel.is_cancelled(gen_id)
    except Exception:
        # Cancellation closes the upstream connection, which surfaces here as a
        # read error; treat it as a normal stop rather than a failure.
        stopped = cancel.is_cancelled(gen_id)
        if not stopped:
            logger.exception(
                "LLM call failed (provider=%s model=%s base_url=%s)",
                profile.get("provider"),
                profile.get("model"),
                profile.get("base_url"),
            )
        if not stopped and not full_answer:
            cancel.release(gen_id)
            yield {
                "type": "done",
                "answer": "调用大模型失败，请检查模型配置后重试。",
                "sources": sources,
                "chunks_used": len(sources),
                "retrieved": _chunk_details(chunks),
                "stopped": False,
                "error": True,
            }
            return
    finally:
        cancel.release(gen_id)

    yield {
        "type": "done",
        "answer": full_answer,
        "sources": sources,
        "chunks_used": len(sources),
        "retrieved": _chunk_details(chunks),
        "stopped": stopped,
    }
