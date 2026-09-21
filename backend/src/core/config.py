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
    # BAAI/bge-m3, not the previous paraphrase-multilingual-MiniLM-L12-v2: a
    # general-purpose similarity model, not retrieval-optimized. bge-m3 is
    # trained specifically for retrieval, covers Vietnamese as part of its
    # 100+ language training, and consistently ranks at the top of
    # multilingual retrieval benchmarks (MIRACL/MTEB) — the strongest
    # available choice given no well-validated Vietnamese-medical-specific
    # embedding model exists (English medical models like PubMedBERT don't
    # transfer to Vietnamese text at all).
    #
    # !!! CHANGING THIS BREAKS RETRIEVAL UNTIL YOU RE-INGEST !!! The existing
    # ChromaDB collection holds 384-dim vectors from the old model; bge-m3
    # produces 1024-dim vectors. Querying old vectors with a new-dimension
    # query embedding does not silently degrade — it errors outright. Do not
    # restart the API server after pulling this change without first running
    # `python -m src.rag.ingest --config config/rag_config.yaml --rebuild`
    # (which will also need `python scripts/build_bm25_index.py` re-run after,
    # same as any re-ingest — see src/rag/bm25_index.py).
    embedding_model: str = "BAAI/bge-m3"
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
    # 40, not the previous 20: widens the vector-search candidate pool the
    # reranker chooses from, to catch borderline matches that wouldn't have
    # made a narrower top-20 cut. Chosen as a middle ground — doubling the
    # reranker's per-query cost (~3.4s -> observed ~doubling) rather than 5x
    # (fetch_k=100) since there's no confirmed case needing more than this.
    retrieval_fetch_k: int = 40
    retrieval_vector_weight: float = 0.6
    retrieval_bm25_weight: float = 0.4
    retrieval_rerank_enabled: bool = True
    retrieval_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_rerank_fetch_k: int = 40
    retrieval_rerank_max_length: int = 512
    retrieval_rerank_batch_size: int = 16
    # Whole-corpus BM25 index (src/rag/bm25_index.py), built by
    # scripts/build_bm25_index.py. Retriever silently skips it if this path
    # doesn't exist yet, falling back to the candidate-only BM25 rerank.
    retrieval_bm25_index_path: str = "data/bm25_index"

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
    # 1200/100, not the previous 512/64: this corpus's records are often long
    # articles (median ~9.7k chars, some 400k+); at 512 chars they fragment
    # into so many chunks per record that retrieval's top-k ends up dominated
    # by several chunks of one weakly-relevant document. See
    # src/rag/chunker.py's TextChunker docstring for the measurements behind
    # this. Changing these only affects the NEXT `python -m src.rag.ingest
    # --rebuild` — already-ingested vectors keep whatever chunking produced
    # them until re-ingested.
    chunk_size: int = 1200
    chunk_overlap: int = 100
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
