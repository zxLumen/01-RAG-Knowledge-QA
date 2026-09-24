from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_base"
    import_db_path: str = "./qdrant_data/imports.db"

    dense_embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    ollama_model: str = "qwen2.5"
    ollama_base_url: str = "http://localhost:11434"

    # LLM provider defaults (overridden by qdrant_data/llm_config.json)
    llm_provider: str = "ollama"
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

    chunk_size: int = 500
    chunk_overlap: int = 75

    top_k: int = 20
    chunks_per_file: int = 3
    rerank_top_k: int = 5
    rrf_k: int = 60


settings = Settings()
