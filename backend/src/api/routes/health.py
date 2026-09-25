"""Health and metrics endpoints."""

from fastapi import APIRouter

from src.api.deps import get_pipeline
from src.api.schemas import HealthResponse, StatsResponse
from src.core.metrics import metrics_endpoint

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check health of all pipeline components."""
    pipeline = get_pipeline()

    try:
        pipeline._lazy_init()
        health = pipeline.check_health()
        stats = pipeline.get_stats()
        components = health.get("components", {})

        all_ok = health.get("status") == "healthy"
        any_error = any("error" in str(v).lower() for v in components.values())

        if all_ok:
            status = "healthy"
        elif any_error:
            status = "unhealthy"
        else:
            status = "degraded"

        return HealthResponse(
            status=status,
            version="1.0.0",
            components=components,
            stats=stats,
        )
    except Exception:
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            components={"pipeline": "not initialized"},
            stats={},
        )


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    """Get pipeline statistics."""
    pipeline = get_pipeline()
    pipeline._lazy_init()
    raw_stats = pipeline.get_stats()

    return StatsResponse(
        initialized=raw_stats.get("initialized", False),
        embedding_model=raw_stats.get("embedding_model", ""),
        vector_store_type=raw_stats.get("vector_store_type", ""),
        collection=raw_stats.get("collection", ""),
        document_count=raw_stats.get("document_count", 0),
        generation_model=raw_stats.get("generation_model", ""),
        retrieval_top_k=raw_stats.get("retrieval_top_k", 5),
        generator_mode=raw_stats.get("generator_mode", "local"),
    )


@router.get("/metrics")
def get_metrics():
    """Prometheus metrics endpoint."""
    return metrics_endpoint()
