from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.config import settings


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default_factory=lambda: settings.top_k)
    collection: Optional[str] = None
    session_id: Optional[str] = None


class QueryCancelRequest(BaseModel):
    session_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


class IngestRequest(BaseModel):
    path: str = "data"
    paths: Optional[list[str]] = None
    recreate: bool = False
    delete_missing: bool = True


class IngestResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    error: Optional[str] = None


class IngestErrorResponse(BaseModel):
    status: str
    error: str


class IngestCancelRequest(BaseModel):
    run_id: Optional[float] = None


class StatusResponse(BaseModel):
    collection: str
    points_count: Optional[int] = None
    status: str


class ConfigResponse(BaseModel):
    embedding_model: str
    embedding_provider: str
    embedding_dim: int
    llm_model: str
    llm_provider: str
    llm_protocol: str
    llm_base_url: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    qdrant_url: str


class ModelInfo(BaseModel):
    name: str
    size: Optional[int] = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    current: str


class ModelSelectRequest(BaseModel):
    model: str


class LLMProfileRequest(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    provider: str = "custom"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = ""
    temperature: Optional[float] = None


class LLMEmbeddingRequest(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    provider: str = "ollama"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = ""
    dim: Optional[int] = None


class LLMTestRequest(BaseModel):
    id: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    target: str = "chat"


class LLMActiveRequest(BaseModel):
    id: str


class LLMProfilePublic(BaseModel):
    id: str
    name: str
    provider: str
    protocol: str
    base_url: str
    model: str
    temperature: float = 0.0
    dim: Optional[int] = None
    has_key: bool = False
    key_hint: str = ""


class LLMConfigResponse(BaseModel):
    active_id: str
    profiles: list[LLMProfilePublic]
    embedding_active_id: str
    embedding_profiles: list[LLMProfilePublic]


class ProvidersResponse(BaseModel):
    providers: list[dict]
    embedding_dims: dict


class ImportFile(BaseModel):
    filename: str
    rel_path: str
    status: str
    chunk_count: int
    file_size: int
    file_md5: Optional[str] = None


class ImportSession(BaseModel):
    id: int
    ts: str
    path: str
    recreate: bool
    documents: int
    chunks: int
    status: str
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    total_files: int = 0
    total_chunks: int = 0


class ImportSessionDetail(ImportSession):
    files: list[ImportFile]
    snapshot_files: list[ImportFile] = []


class ImportListResponse(BaseModel):
    count: int
    sessions: list[ImportSession]


class CollectionSwitchRequest(BaseModel):
    name: str


class CollectionRenameRequest(BaseModel):
    name: str
    new_name: str


class PasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str


class CollectionsResponse(BaseModel):
    current: str
    collections: list[str]
