from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    qdrant_url: str = "http://localhost:6333"
    # Optional API key when QDRANT_URL points at a Qdrant server.
    qdrant_api_key: str = ""
    qdrant_collection: str = "knowledge_base"
    import_db_path: str = "./qdrant_data/imports.db"
    chat_db_path: str = "./qdrant_data/chats.db"

    dense_embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    ollama_model: str = "qwen2.5"
    ollama_base_url: str = "http://localhost:11434"

    # LLM provider defaults (overridden by qdrant_data/llm_config.json)
    llm_provider: str = "zx-gateway"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.0

    # Embedding provider defaults
    embedding_provider: str = "ollama"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""

    # Demo protection: when set, write operations on /api/llm/* require this token
    admin_token: str = ""
    # Simple per-IP rate limit for query endpoints (requests per minute, 0 = off)
    rate_limit_per_minute: int = 0
    # Visitor isolation (per-cookie data dir + collection). Enable for public demos.
    demo_mode: bool = False

    # Visitor knowledge-base size caps (points/chunks). Only apply to visitors,
    # never to the admin's own collections. The global cap counts all visitor
    # collections; when exceeded, least-recently-active visitors are evicted
    # (their collection only) to make room.
    # 5000 is sized so any single shared doc imports (largest is ~3000 chunks),
    # which still leaves a visitor well under half of the global budget.
    visitor_max_points: int = 5000
    visitor_global_max_points: int = 150000
    # Max new visitor identities minted per IP per hour (0 = unlimited).
    visitor_mint_per_hour: int = 20

    # Retrieval tuning defaults. Env vars set these; the runtime values are
    # held in qdrant_data/retrieval_config.json via src.retrieval.config.
    chunk_size: int = 500
    chunk_overlap: int = 75

    top_k: int = 20
    chunks_per_file: int = 3
    # Post-selection cap on chunks handed to the LLM (0 = no cap).
    rerank_top_k: int = 0
    rrf_k: int = 60

    # Hybrid retrieval (dense + sparse/BM25 fused with weighted RRF).
    hybrid_enabled: bool = True
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    # Per-channel recall pool = max(top_k * candidate_multiplier, candidate_min).
    candidate_multiplier: int = 4
    candidate_min: int = 24
    # Legacy keyword channel (used as another RRF channel).
    keyword_boost: float = 1.1
    keyword_limit: int = 5
    # Context assembly budget (characters) and optional dense score floor.
    max_context_chars: int = 4000
    min_score: float = 0.0


settings = Settings()
