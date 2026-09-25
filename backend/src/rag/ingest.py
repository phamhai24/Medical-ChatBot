"""Script to ingest medical data into the vector store"""

import argparse
import logging
from pathlib import Path

from src.utils.config_loader import load_config
from src.rag.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest medical data into RAG vector store")
    parser.add_argument(
        "--config",
        type=str,
        default="config/rag_config.yaml",
        help="Path to RAG config file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/data.json",
        help="Path to data file"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index from scratch (clear existing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for embedding"
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Hide progress bar"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    config = load_config(str(config_path))
    logger.info(f"Loaded config from {config_path}")

    # Initialize pipeline
    pipeline = RAGPipeline(config)

    # Ingest data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    logger.info(f"Starting ingestion of {data_path}")
    stats = pipeline.ingest(
        data_path=str(data_path),
        batch_size=args.batch_size,
        rebuild=args.rebuild,
        show_progress=not args.no_progress,
    )

    logger.info("=" * 50)
    logger.info("✅ Ingestion Summary:")
    logger.info(f"   Records processed: {stats['total_records']}")
    logger.info(f"   Chunks created:   {stats['total_chunks']}")
    logger.info(f"   Avg chunk length: {stats['avg_chunk_length']:.0f} chars")
    logger.info(f"   Embedding model:  {stats['embedding_model']}")
    logger.info(f"   Embedding dim:   {stats['embedding_dimension']}")
    logger.info(f"   Vector store:     {stats['vector_store_type']}")
    logger.info(f"   Documents indexed: {stats['documents_indexed']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
