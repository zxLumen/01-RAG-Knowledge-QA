# RAG Knowledge QA — 运行时镜像
# app 本体很轻:dense embedding 与 LLM 均通过 OLLAMA_BASE_URL 调用外部 Ollama,
# 本地只用 FastEmbed 的 bm25(sparse, 极小)。故镜像不含大模型权重。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    HF_HOME=/models/hf

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY static ./static
COPY run_server.py rag.py ./

RUN pip install --upgrade pip && pip install .

# 预下载 sparse(bm25)模型到镜像内缓存,避免运行时联网拉取
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/status || exit 1

# 本地 qdrant 模式重启后可能残留 .lock,启动前清理
CMD ["sh", "-c", "rm -f /app/qdrant_data/.lock; exec python run_server.py"]
