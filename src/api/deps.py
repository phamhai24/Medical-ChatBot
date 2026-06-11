"""Dependency injection for FastAPI routes."""

from typing import Optional

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
