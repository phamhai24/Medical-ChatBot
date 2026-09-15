"""Pydantic Settings - type-safe configuration from env vars and YAML."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Paths ────────────────────────────────────────────────────────────────
    project_root: Path = Field(default=ROOT)
    config_path: Path = Field(default=ROOT / "config" / "rag_config.yaml")

    # ─── API ────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_workers: int = 1
    api_cors_origins: list[str] = ["http://localhost:5173"]
    api_warmup_enabled: bool = True
    api_warmup_query: str = "kiểm tra sức khỏe hệ thống"
    admin_api_key: Optional[str] = None

    # ─── Redis ──────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_enabled: bool = False  # Set True to enable session + cache

    # ─── RAG - Embedding ───────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "auto"  # auto, cpu, cuda
    embedding_batch_size: int = 32
    embedding_normalize: bool = True
    embedding_max_length: int = 512

    # ─── RAG - Vector Store ─────────────────────────────────────────────────
    vector_store_type: str = "chroma"
    vector_store_path: str = "data/vectorstore"
    vector_store_collection: str = "medical_qa"
    vector_search_type: str = "similarity"
    vector_fetch_k: int = 20
    vector_lambda_mult: float = 0.5

    # ─── RAG - Retrieval ────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.3
    retrieval_fetch_k: int = 20
    retrieval_vector_weight: float = 0.6
    retrieval_bm25_weight: float = 0.4

    # ─── RAG - Generation ──────────────────────────────────────────────────
    generator_mode: str = "local"  # local, api
    generator_model: str = "Qwen/Qwen2.5-7B-Instruct"
    generator_max_tokens: int = 512
    generator_temperature: float = 0.3
    generator_top_p: float = 0.9
    generator_top_k: int = 50
    generator_do_sample: bool = True
    generator_repetition_penalty: float = 1.1

    # ─── API Generator (Groq/OpenAI) ─────────────────────────────────────────
    api_generator_provider: str = "groq"  # groq, openai, anthropic
    api_generator_model: str = "llama-3.3-70b-versatile"
    api_generator_base_url: Optional[str] = None
    api_generator_api_key: Optional[str] = None
    api_generator_max_retries: int = 2
    api_generator_retry_backoff: float = 1.0
    api_generator_retry_max_backoff: float = 8.0
    api_generator_circuit_breaker_threshold: int = 3
    api_generator_circuit_breaker_cooldown: float = 30.0
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # ─── RAGAS Evaluation ──────────────────────────────────────────────────
    ragas_enabled: bool = False
    ragas_llm_api_key: Optional[str] = None
    ragas_embedding_api_key: Optional[str] = None

    # ─── Logging ────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_json: bool = True
    log_file: str = "logs/app.log"
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"

    # ─── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_min_length: int = 50

    # ─── Rate Limiting ─────────────────────────────────────────────────────
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 30
    rate_limit_window: int = 60  # seconds

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_cuda(self) -> bool:
        if self.embedding_device == "auto":
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
        return self.embedding_device == "cuda"

    @model_validator(mode="after")
    def resolve_api_generator_key(self) -> "Settings":
        key_map = {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }
        if self.api_generator_api_key is None:
            self.api_generator_api_key = key_map.get(self.api_generator_provider)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
