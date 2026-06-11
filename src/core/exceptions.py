"""Custom exceptions and global exception handlers for FastAPI."""

from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    """Base application exception."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class PipelineNotInitializedError(AppException):
    """Raised when RAG pipeline hasn't been initialized."""

    def __init__(self, message: str = "RAG pipeline not initialized"):
        super().__init__(message, status_code=503)


class VectorStoreError(AppException):
    """Raised when vector store operations fail."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(f"Vector store error: {message}", status_code=500, details=details)


class EmbeddingError(AppException):
    """Raised when embedding operations fail."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(f"Embedding error: {message}", status_code=500, details=details)


class GenerationError(AppException):
    """Raised when LLM generation fails."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(f"Generation error: {message}", status_code=500, details=details)


class IngestionError(AppException):
    """Raised when data ingestion fails."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(f"Ingestion error: {message}", status_code=500, details=details)


class DataNotFoundError(AppException):
    """Raised when required data file is not found."""

    def __init__(self, message: str):
        super().__init__(f"Data not found: {message}", status_code=404)


class ValidationError(AppException):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(f"Validation error: {message}", status_code=422, details=details)


class RateLimitExceededError(AppException):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


def setup_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        from src.core.logging import get_correlation_id

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "correlation_id": get_correlation_id(),
                "details": exc.details,
            },
            headers={"X-Correlation-ID": get_correlation_id()},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        from src.core.logging import get_correlation_id, logger

        logger.exception(f"Unhandled exception: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "correlation_id": get_correlation_id(),
                "details": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
            headers={"X-Correlation-ID": get_correlation_id()},
        )
