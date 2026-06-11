"""Data ingestion pipeline."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full data ingestion pipeline:
    load -> chunk -> embed -> store

    Supports parallel embedding for faster processing.
    """

    def __init__(
        self,
        pipeline,
        batch_size: int = 100,
        parallel_workers: int = 4,
    ):
        """
        Args:
            pipeline: RAGPipeline instance
            batch_size: Batch size for embedding
            parallel_workers: Number of parallel workers for embedding
        """
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.parallel_workers = parallel_workers
        self._chunker = None

    def run(
        self,
        data_path: str,
        rebuild: bool = False,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """
        Run the full ingestion pipeline.

        Args:
            data_path: Path to JSON data file
            rebuild: Whether to clear existing index
            show_progress: Show progress bars

        Returns:
            Ingestion statistics
        """
        from tqdm import tqdm

        start_time = time.perf_counter()

        # Load data
        logger.info(f"Loading data from {data_path}")
        from src.ingestion.loaders.json_loader import JSONLoader

        loader = JSONLoader(data_path)
        data = loader.load()
        total_records = len(data)

        # Initialize components
        self.pipeline._lazy_init()

        # Clear if rebuilding
        if rebuild:
            logger.info("Clearing existing index...")
            self.pipeline.vector_store.clear()

        # Chunk
        if self._chunker is None:
            from src.rag.chunker import TextChunker

            chunk_config = self.pipeline.config.get("rag", {}).get("data", {})
            self._chunker = TextChunker(
                chunk_size=chunk_config.get("chunk_size", 512),
                chunk_overlap=chunk_config.get("chunk_overlap", 64),
                min_chunk_length=chunk_config.get("min_chunk_length", 50),
            )

        logger.info("Chunking documents...")
        documents = self._chunker.chunk_qa_pairs(data, include_question=True)
        chunk_stats = self._chunker.get_chunk_stats(documents)
        logger.info(f"Created {chunk_stats['total_chunks']} chunks")

        # Load embedder
        self.pipeline.embedder.load()
        dim = self.pipeline.embedder.get_embedding_dimension()

        # Batch embedding and indexing
        texts = [doc["text"] for doc in documents]
        metadatas = [
            {
                "question": doc["question"],
                "chunk_index": doc["chunk_index"],
                "total_chunks": doc["total_chunks"],
                "source": doc["source"],
            }
            for doc in documents
        ]
        ids = [doc["chunk_id"] for doc in documents]

        logger.info("Generating embeddings and indexing...")

        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        progress = tqdm(range(n_batches), disable=not show_progress, desc="Embedding")

        for i in progress:
            batch_start = i * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(texts))

            batch_texts = texts[batch_start:batch_end]
            batch_embeddings = self.pipeline.embedder.embed(batch_texts)

            self.pipeline.vector_store.add_documents(
                texts=batch_texts,
                embeddings=batch_embeddings,
                metadatas=metadatas[batch_start:batch_end],
                ids=ids[batch_start:batch_end],
            )

            progress.set_postfix({
                "indexed": batch_end,
                "total": len(texts),
            })

        duration = time.perf_counter() - start_time

        stats = {
            "total_records": total_records,
            "total_chunks": len(documents),
            "avg_chunk_length": chunk_stats["avg_length"],
            "embedding_model": self.pipeline._embedding_config.get("model_name"),
            "embedding_dimension": dim,
            "vector_store_type": self.pipeline._vector_config.get("type"),
            "documents_indexed": self.pipeline.vector_store.count(),
            "duration_seconds": round(duration, 2),
        }

        logger.info(f"Ingestion complete: {stats}")
        return stats
