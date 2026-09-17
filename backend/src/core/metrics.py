"""Prometheus metrics for observability."""

import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# ─── Request Metrics ─────────────────────────────────────────────────────────

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ─── RAG Pipeline Metrics ─────────────────────────────────────────────────────

rag_queries_total = Counter(
    "rag_queries_total",
    "Total RAG queries processed",
    ["status"],  # success, no_results, error
)

rag_retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval step latency",
    ["search_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

rag_generation_latency_seconds = Histogram(
    "rag_generation_latency_seconds",
    "Generation step latency",
    ["generator_mode"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

rag_generation_tokens = Histogram(
    "rag_generation_tokens_total",
    "Tokens generated per request",
    ["generator_mode"],
    buckets=(10, 50, 100, 200, 400, 800, 1600),
)

rag_sources_retrieved = Histogram(
    "rag_sources_retrieved",
    "Number of sources retrieved per query",
    buckets=(1, 2, 3, 5, 10, 20),
)

# ─── Ingestion Metrics ────────────────────────────────────────────────────────

ingestion_records_total = Counter(
    "ingestion_records_total",
    "Total records ingested",
    ["status"],  # success, failed
)

ingestion_chunks_created = Gauge(
    "ingestion_chunks_created",
    "Total chunks created",
)

# ─── System Metrics ──────────────────────────────────────────────────────────

vectorstore_document_count = Gauge(
    "vectorstore_document_count",
    "Number of documents in vector store",
)

# ─── Utility Functions ────────────────────────────────────────────────────────


def metrics_endpoint() -> Response:
    """Generate Prometheus metrics response."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def track_request_metrics(method: str, endpoint: str, status: int, duration: float):
    """Track HTTP request metrics."""
    http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)


def track_rag_metrics(
    status: str,
    retrieval_latency: float,
    generation_latency: float,
    tokens: int,
    sources_count: int,
    generator_mode: str,
    search_type: str,
):
    """Track RAG pipeline metrics."""
    rag_queries_total.labels(status=status).inc()
    if retrieval_latency > 0:
        rag_retrieval_latency_seconds.labels(search_type=search_type).observe(retrieval_latency)
    if generation_latency > 0:
        rag_generation_latency_seconds.labels(generator_mode=generator_mode).observe(generation_latency)
    if tokens > 0:
        rag_generation_tokens.labels(generator_mode=generator_mode).observe(tokens)
    if sources_count > 0:
        rag_sources_retrieved.observe(sources_count)


def timed(metric: Histogram, labels: dict[str, str] | None = None):
    """Decorator to time function execution."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapper
    return decorator
