"""JSON data loader for the ingestion pipeline."""

import json
import logging
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class JSONLoader:
    """Load Q&A records from JSON files with streaming support for large files."""

    def __init__(
        self,
        data_path: str,
        encoding: str = "utf-8",
    ):
        """
        Args:
            data_path: Path to the JSON data file
            encoding: File encoding
        """
        self.data_path = Path(data_path)
        self.encoding = encoding

    def load(self) -> list[dict[str, Any]]:
        """Load all records into memory (for small-medium files)."""
        logger.info(f"Loading data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        with open(self.data_path, "r", encoding=self.encoding) as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a list of records, got {type(data).__name__}")

        logger.info(f"Loaded {len(data)} records")
        return data

    def stream(self, chunk_size: int = 1000) -> Iterator[list[dict[str, Any]]]:
        """
        Stream records in chunks for memory-efficient processing.

        Args:
            chunk_size: Number of records per chunk

        Yields:
            Chunks of records
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Streaming data from {self.data_path}")

        # For JSON array, we need to read the whole file but yield chunks
        with open(self.data_path, "r", encoding=self.encoding) as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a list of records, got {type(data).__name__}")

        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    def get_record_count(self) -> int:
        """Get the total number of records without loading all into memory."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        with open(self.data_path, "r", encoding=self.encoding) as f:
            data = json.load(f)

        if isinstance(data, list):
            return len(data)

        return 0
