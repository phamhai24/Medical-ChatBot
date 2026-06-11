"""Text chunking for medical Q&A data"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Split documents into smaller chunks for embedding.
    Optimized for medical Q&A pairs with variable-length answers.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_length: int = 50,
        separator: str = "\n"
    ):
        """
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
            min_chunk_length: Minimum chunk length to keep
            separator: Separator to split text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.separator = separator

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if not text or len(text) < self.min_chunk_length:
            return [text] if text else []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size

            # Try to break at separator near end of chunk
            if end < text_len:
                # Look for paragraph/line breaks first
                split_point = text.rfind("\n\n", start + int(self.chunk_size * 0.7), end)
                if split_point > start + int(self.chunk_size * 0.5):
                    end = split_point + 2
                else:
                    # Fallback: find sentence boundary
                    split_point = text.rfind(". ", start + int(self.chunk_size * 0.7), end)
                    if split_point > start + int(self.chunk_size * 0.5):
                        end = split_point + 1

            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_length:
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start >= text_len - self.chunk_overlap:
                # Add final chunk
                final_chunk = text[start:].strip()
                if len(final_chunk) >= self.min_chunk_length:
                    chunks.append(final_chunk)
                break

        return chunks

    def chunk_qa_pairs(
        self,
        data: List[Dict[str, Any]],
        include_question: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk Q&A pairs from medical dataset.

        Args:
            data: List of {"question": ..., "answer": ...} records
            include_question: Whether to include question in chunk context

        Returns:
            List of chunked documents with metadata
        """
        documents = []
        doc_id = 0

        for record in data:
            question = record.get("question", "")
            answer = record.get("answer", "")

            if not answer:
                continue

            # Chunk the answer (answers are typically longer)
            answer_chunks = self.chunk_text(answer)

            for chunk_idx, chunk_text in enumerate(answer_chunks):
                chunk_id = f"{doc_id}_{chunk_idx}"

                if include_question:
                    # Prepend question for better context
                    combined_text = f"Câu hỏi: {question}\n\nTrả lời: {chunk_text}"
                else:
                    combined_text = chunk_text

                documents.append({
                    "chunk_id": chunk_id,
                    "text": combined_text,
                    "question": question,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(answer_chunks),
                    "source": "medical_qa"
                })

            doc_id += 1

        logger.info(f"Created {len(documents)} chunks from {doc_id} Q&A pairs")
        return documents

    def get_chunk_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunks."""
        if not documents:
            return {}

        lengths = [len(doc["text"]) for doc in documents]

        return {
            "total_chunks": len(documents),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_characters": sum(lengths),
        }
