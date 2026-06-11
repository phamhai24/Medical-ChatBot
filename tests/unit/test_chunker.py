"""Unit tests for the TextChunker."""

import pytest
from src.rag.chunker import TextChunker


class TestTextChunker:
    def test_chunk_text_short_text(self):
        """Short text should return as single chunk."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64, min_chunk_length=50)
        text = "Triệu chứng bệnh tiểu đường."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_long_text(self):
        """Long text should be split into multiple chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_length=30)
        text = "A" * 500
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_text_empty(self):
        """Empty text should return empty list."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text(None) == []

    def test_chunk_qa_pairs(self, sample_qa_data):
        """Q&A pairs should be chunked with question prepended."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64, min_chunk_length=50)
        documents = chunker.chunk_qa_pairs(sample_qa_data, include_question=True)

        assert len(documents) == len(sample_qa_data)
        for doc in documents:
            assert "chunk_id" in doc
            assert "text" in doc
            assert "question" in doc
            assert "Câu hỏi:" in doc["text"]
            assert "Trả lời:" in doc["text"]

    def test_chunk_qa_pairs_without_question(self, sample_qa_data):
        """Q&A pairs can be chunked without question prepended."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64, min_chunk_length=50)
        documents = chunker.chunk_qa_pairs(sample_qa_data, include_question=False)

        assert len(documents) == len(sample_qa_data)
        for doc in documents:
            assert "Câu hỏi:" not in doc["text"]

    def test_chunk_stats(self, sample_qa_data):
        """Chunk stats should be computed correctly."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64, min_chunk_length=50)
        documents = chunker.chunk_qa_pairs(sample_qa_data)
        stats = chunker.get_chunk_stats(documents)

        assert "total_chunks" in stats
        assert stats["total_chunks"] == len(documents)
        assert "avg_length" in stats
        assert "min_length" in stats
        assert "max_length" in stats

    def test_chunk_ids_unique(self, sample_qa_data):
        """All chunk IDs should be unique."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64, min_chunk_length=50)
        documents = chunker.chunk_qa_pairs(sample_qa_data)
        ids = [doc["chunk_id"] for doc in documents]
        assert len(ids) == len(set(ids))
