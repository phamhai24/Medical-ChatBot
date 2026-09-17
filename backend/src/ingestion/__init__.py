"""Ingestion package."""

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.loaders.json_loader import JSONLoader

__all__ = ["IngestionPipeline", "JSONLoader"]
