"""Dependency injection for FastAPI routes."""

from typing import Optional

from fastapi import Depends, Header, HTTPException

from src.core.config import get_settings
from src.core.logging import logger


# ─── Global Pipeline ──────────────────────────────────────────────────────────

_pipeline: Optional[object] = None


def get_pipeline():
    """Get or create the RAG pipeline singleton (lazy initialization)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from src.rag.pipeline import RAGPipeline
    from src.utils.config_loader import load_config

    config = load_config()
    _pipeline = RAGPipeline(config)
    logger.info("RAG pipeline initialized via dependency injection")
    return _pipeline


def reset_pipeline():
    """Reset the pipeline (useful for testing or re-initialization)."""
    global _pipeline
    _pipeline = None


# ─── Settings ─────────────────────────────────────────────────────────────────

def get_settings_dep():
    """Get application settings."""
    return get_settings()


# ─── Admin Auth ───────────────────────────────────────────────────────────────

_admin_key_warning_logged = False


def verify_admin_key(
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
    settings=Depends(get_settings_dep),
) -> None:
    """Require X-Admin-Key to match ADMIN_API_KEY when it's configured."""
    global _admin_key_warning_logged

    if not settings.admin_api_key:
        if not _admin_key_warning_logged:
            logger.warning("ADMIN_API_KEY not set - /api/v1/admin/* is unauthenticated")
            _admin_key_warning_logged = True
        return

    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")
