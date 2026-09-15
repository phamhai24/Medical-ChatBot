"""FastAPI application entry point."""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import admin, chat, health, session
from src.api.deps import get_pipeline
from src.api.schemas import ErrorResponse
from src.core.config import get_settings
from src.core.exceptions import setup_exception_handlers
from src.core.logging import add_correlation_id, configure_from_settings, logger
from src.core.metrics import track_request_metrics
from src.core.redis_client import close_redis, init_redis


# ─── Middleware ───────────────────────────────────────────────────────────────

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        add_correlation_id(correlation_id)

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        import time
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # Normalize path for metrics (avoid high cardinality)
        path = request.url.path
        if path.startswith("/api/v1/chat"):
            path = "/api/v1/chat"
        elif path.startswith("/api/v1/admin"):
            path = "/api/v1/admin"

        track_request_metrics(
            method=request.method,
            endpoint=path,
            status=response.status_code,
            duration=duration,
        )
        return response


# ─── App Factory ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Medical RAG Chatbot API...")

    # Init Redis if enabled
    if settings.redis_enabled:
        try:
            init_redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
            )
            logger.info("Redis client initialized")
        except Exception as e:
            logger.warning(f"Redis connection failed (continuing without Redis): {e}")

    if settings.api_warmup_enabled:
        try:
            logger.info("Warming up RAG pipeline...")
            pipeline = get_pipeline()
            warmup_stats = pipeline.warm_up(settings.api_warmup_query)
            logger.info(f"RAG pipeline warm-up complete: {warmup_stats}")
        except Exception as e:
            logger.exception(f"RAG pipeline warm-up failed: {e}")
            raise

    yield

    # Shutdown
    close_redis()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    # Setup logging first
    configure_from_settings(settings)

    app = FastAPI(
        title="Medical RAG Chatbot API",
        description="API cho Chatbot Y tế sử dụng RAG pipeline với evaluation framework",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(RequestMetricsMiddleware)

    # Exception handlers
    setup_exception_handlers(app)

    # Register routes
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(admin.router)
    app.include_router(session.router)

    # Root
    @app.get("/", tags=["Root"])
    def root():
        return {
            "name": "Medical RAG Chatbot API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
        }

    logger.info("FastAPI app created successfully")
    return app


app = create_app()


def run() -> None:
    """Run the versioned API when invoked with `python -m src.api.main`."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers if not settings.api_reload else 1,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
