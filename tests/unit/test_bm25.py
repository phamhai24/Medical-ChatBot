"""Unit tests for the BM25 hybrid retriever."""

import pytest
from src.rag.hybrid_search import BM25Retriever, HybridRetriever


class TestBM25Retriever:
    def test_build_index(self):
        """BM25 index should be built correctly."""
        retriever = BM25Retriever()
        docs = [
            {"id": "1", "text": "Triệu chứng bệnh tiểu đường là gì?"},
            {"id": "2", "text": "Cách phòng ngừa cao huyết áp?"},
            {"id": "3", "text": "Vitamin C tốt cho sức khỏe."},
        ]
        retriever.build_index(docs)
        assert hasattr(retriever, "_documents")
        assert retriever._N == 3

    def test_search_returns_scored_docs(self):
        """BM25 search should return documents with scores."""
        retriever = BM25Retriever()
        docs = [
            {"id": "1", "text": "Triệu chứng bệnh tiểu đường là gì?"},
            {"id": "2", "text": "Cách phòng ngừa cao huyết áp?"},
            {"id": "3", "text": "Vitamin C tốt cho sức khỏe."},
        ]
        retriever.build_index(docs)
        results = retriever.search("tiểu đường triệu chứng")

        assert len(results) > 0
        assert all("bm25_score" in r for r in results)
        assert results[0]["id"] == "1"  # Most relevant

    def test_search_empty_index(self):
        """BM25 search on empty index should return empty list."""
        retriever = BM25Retriever()
        results = retriever.search("test query")
        assert results == []

    def test_scores_descending(self):
        """BM25 scores should be in descending order."""
        retriever = BM25Retriever()
        docs = [
            {"id": "1", "text": "Triệu chứng bệnh tiểu đường là gì?"},
            {"id": "2", "text": "Cách phòng ngừa cao huyết áp?"},
            {"id": "3", "text": "Vitamin C tốt cho sức khỏe."},
        ]
        retriever.build_index(docs)
        results = retriever.search("tiểu đường")
        scores = [r["bm25_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
