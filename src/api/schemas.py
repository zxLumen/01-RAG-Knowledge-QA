from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    chunks_per_file: Optional[int] = None
    collection: Optional[str] = None
    session_id: Optional[str] = None


class RetrieveRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    chunks_per_file: Optional[int] = None
    collection: Optional[str] = None


class RetrievedChunkItem(BaseModel):
    filename: str
    rel_path: str
    text: str
    score: float


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunkItem]
    sources: list[str]
    count: int


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
    chunks_per_file: int
    hybrid_enabled: bool
    rrf_k: int
    rerank_top_k: int
    candidate_multiplier: int
    candidate_min: int
    dense_weight: float
    sparse_weight: float
    keyword_boost: float
    keyword_limit: int
    max_context_chars: int
    min_score: float
    qdrant_url: str


class RetrievalConfigResponse(BaseModel):
    top_k: int
    chunks_per_file: int
    rerank_top_k: int
    rrf_k: int
    hybrid_enabled: bool
    dense_weight: float
    sparse_weight: float
    candidate_multiplier: int
    candidate_min: int
    keyword_boost: float
    keyword_limit: int
    max_context_chars: int
    min_score: float
    chunk_size: int
    chunk_overlap: int


class RetrievalConfigRequest(BaseModel):
    top_k: Optional[int] = None
    chunks_per_file: Optional[int] = None
    rerank_top_k: Optional[int] = None
    rrf_k: Optional[int] = None
    hybrid_enabled: Optional[bool] = None
    dense_weight: Optional[float] = None
    sparse_weight: Optional[float] = None
    candidate_multiplier: Optional[int] = None
    candidate_min: Optional[int] = None
    keyword_boost: Optional[float] = None
    keyword_limit: Optional[int] = None
    max_context_chars: Optional[int] = None
    min_score: Optional[float] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


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


class FileDeleteRequest(BaseModel):
    rels: list[str]


class CollectionRenameRequest(BaseModel):
    name: str
    new_name: str


class PasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str


class UIConfigRequest(BaseModel):
    themes: list[str]
    default_theme: str


class UIConfigResponse(BaseModel):
    themes: list[str]
    default_theme: str


class CollectionsResponse(BaseModel):
    current: str
    collections: list[str]
    shared: list[str] = []


class ShareConfigRequest(BaseModel):
    collections: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)


class ShareConfigResponse(BaseModel):
    collections: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)


class AdminResetRequest(BaseModel):
    admin_token: str
    new_password: str


class ChatSyncRequest(BaseModel):
    session_id: str = Field(min_length=1)
    title: str = ""
    collection: str = ""
    messages: list[dict] = Field(default_factory=list)
    deleted: bool = False


class ChatSessionItem(BaseModel):
    rowid: int
    id: str
    owner: str = ""
    title: str = ""
    collection: str = ""
    messages: list[dict] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    deleted: bool = False


class ChatListResponse(BaseModel):
    owner: str = ""
    sessions: list[ChatSessionItem]


class ChatAdminListResponse(BaseModel):
    count: int
    sessions: list[ChatSessionItem]


class ChatRowRequest(BaseModel):
    rowid: int
