"""Admin endpoints - ingestion and reindexing."""

import time

from fastapi import APIRouter, HTTPException

from src.api.deps import get_pipeline
from src.api.schemas import IngestRequest, IngestResponse, ReindexRequest
from src.core.logging import logger

router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_data(request: IngestRequest):
    """Ingest dữ liệu vào vector store."""
    pipeline = get_pipeline()

    start = time.perf_counter()

    try:
        stats = pipeline.ingest(
            data_path=request.data_path,
            batch_size=request.batch_size,
            rebuild=request.rebuild,
            show_progress=False,
        )

        duration = time.perf_counter() - start

        return IngestResponse(
            success=True,
            total_records=stats.get("total_records", 0),
            total_chunks=stats.get("total_chunks", 0),
            avg_chunk_length=stats.get("avg_chunk_length", 0),
            embedding_model=stats.get("embedding_model", ""),
            embedding_dimension=stats.get("embedding_dimension", 0),
            vector_store_type=stats.get("vector_store_type", ""),
            documents_indexed=stats.get("documents_indexed", 0),
            duration_seconds=round(duration, 2),
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data file not found: {request.data_path}")
    except Exception as e:
        logger.exception(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex")
def reindex(request: ReindexRequest):
    """Xóa và tạo lại vector index."""
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Must set confirm=true to reindex")

    pipeline = get_pipeline()
    pipeline._lazy_init()

    try:
        pipeline.vector_store.clear()
        logger.info("Vector store cleared for reindexing")

        return {"success": True, "message": "Index cleared. Call /ingest to rebuild."}
    except Exception as e:
        logger.exception(f"Reindex error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
