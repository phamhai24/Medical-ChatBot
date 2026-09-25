"""Pydantic request/response schemas for the API."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─── Chat ────────────────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    """One earlier message of the conversation, sent by the client."""
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


_TOPIC_FIELD_DESCRIPTION = (
    "Chủ đề đang trao đổi, lấy từ câu trả lời trước (trường `topic`). "
    "Nếu bỏ trống, server lấy từ lịch sử Redis theo session_id."
)

_HISTORY_FIELD_DESCRIPTION = (
    "Các tin nhắn gần nhất (cũ trước, mới sau), dùng để hiểu câu hỏi nối tiếp. "
    "Nếu bỏ trống, server lấy lịch sử từ Redis theo session_id."
)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Câu hỏi của người dùng")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Số tài liệu cần retrieve")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature cho generation")
    include_sources: bool = Field(True, description="Bao gồm nguồn trích dẫn")
    session_id: Optional[str] = Field(None, description="Session ID cho lịch sử chat")
    history: Optional[list[HistoryMessage]] = Field(
        None, max_length=20, description=_HISTORY_FIELD_DESCRIPTION
    )
    topic: Optional[str] = Field(None, max_length=100, description=_TOPIC_FIELD_DESCRIPTION)


class SourceDocument(BaseModel):
    id: str
    question: str
    score: float
    chunk_index: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    latency_ms: float
    model: str
    top_k: int
    session_id: Optional[str] = None
    topic: Optional[str] = Field(None, description="Chủ đề đang trao đổi; gửi lại ở câu hỏi tiếp theo")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ChatStreamRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(5, ge=1, le=20)
    session_id: Optional[str] = None
    history: Optional[list[HistoryMessage]] = Field(
        None, max_length=20, description=_HISTORY_FIELD_DESCRIPTION
    )
    topic: Optional[str] = Field(None, max_length=100, description=_TOPIC_FIELD_DESCRIPTION)


# ─── Session ─────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # user | assistant
    content: str
    metadata: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]
    total: int


# ─── Admin / Ingestion ───────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    data_path: str = Field("data/processed/data.json", description="Đường dẫn file dữ liệu")
    rebuild: bool = Field(False, description="Xóa index cũ trước khi ingest")
    batch_size: int = Field(100, ge=1, le=1000, description="Batch size cho embedding")


class IngestResponse(BaseModel):
    success: bool
    total_records: int
    total_chunks: int
    avg_chunk_length: float
    embedding_model: str
    embedding_dimension: int
    vector_store_type: str
    documents_indexed: int
    duration_seconds: float


class ReindexRequest(BaseModel):
    confirm: bool = Field(..., description="Xác nhận xóa index cũ")


# ─── Health & System ─────────────────────────────────────────────────────────

class ComponentHealth(BaseModel):
    status: str  # ok, error, degraded
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str  # healthy, degraded, unhealthy
    version: str
    components: dict[str, str]
    stats: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StatsResponse(BaseModel):
    initialized: bool
    embedding_model: str
    vector_store_type: str
    collection: str
    document_count: int
    generation_model: str
    retrieval_top_k: int
    generator_mode: str


# ─── Error ───────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    correlation_id: Optional[str] = None
    details: Optional[dict[str, Any]] = None
