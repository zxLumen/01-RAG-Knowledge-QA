# RAG Knowledge QA — 运行时镜像
# app 本体很轻:dense embedding 与 LLM 可走外部 API(推荐)或 Ollama,
# 本地只用 FastEmbed 的 bm25(sparse, 极小)。故镜像不含大模型权重。
FROM python:3.12-slim

ARG APP_BUILD=dev
ENV APP_BUILD=$APP_BUILD

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    HF_HOME=/models/hf

# 可选:HF 受限环境(如中国大陆)构建时改用镜像站:
#   docker build --build-arg HF_ENDPOINT=https://hf-mirror.com .
ARG HF_ENDPOINT=https://huggingface.co
ENV HF_ENDPOINT=${HF_ENDPOINT}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gosu \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY static ./static
# 只读样例随镜像内置(仓库 data/ 被 gitignore,故单独放 samples/)
COPY samples ./data/samples
COPY run_server.py rag.py ./
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN pip install --upgrade pip && pip install .

# 预下载 sparse(bm25)模型到镜像内缓存,避免运行时联网拉取
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')" \
    && useradd -m -u 10001 appuser \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R appuser:appuser /models

EXPOSE 8000

# 用 /api/version 做健康检查:无副作用(/api/status 在演示模式下会新建访客)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/version || exit 1

# entrypoint 修复卷属主后以非 root 运行
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
