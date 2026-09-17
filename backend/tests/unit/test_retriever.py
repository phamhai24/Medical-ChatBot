"""Unit tests for the Retriever."""

import pytest
from src.rag.retriever import Retriever


class TestRetriever:
    def test_retrieve_empty_results(self, mock_embedder, mock_vector_store):
        """Should return empty list when no docs indexed."""
        retriever = Retriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            top_k=5,
        )
        results = retriever.retrieve("bệnh gì đó")
        assert results == []

    def test_retrieve_with_results(self, mock_embedder, mock_vector_store, sample_qa_data):
        """Should return relevant documents."""
        # First add some documents
        texts = [f"Câu hỏi: {qa['question']}\n\nTrả lời: {qa['answer']}" for qa in sample_qa_data]
        embeddings = mock_embedder.embed(texts)
        mock_vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=[{"question": qa["question"]} for qa in sample_qa_data],
            ids=[qa["id"] for qa in sample_qa_data],
        )

        retriever = Retriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            top_k=5,
        )

        results = retriever.retrieve("triệu chứng tiểu đường")
        assert len(results) > 0
        assert "text" in results[0]
        assert "distance" in results[0]

    def test_retrieve_top_k(self, mock_embedder, mock_vector_store, sample_qa_data):
        """Should respect top_k parameter."""
        texts = [f"Câu hỏi: {qa['question']}\n\nTrả lời: {qa['answer']}" for qa in sample_qa_data]
        embeddings = mock_embedder.embed(texts)
        mock_vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=[{"question": qa["question"]} for qa in sample_qa_data],
            ids=[qa["id"] for qa in sample_qa_data],
        )

        retriever = Retriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            top_k=2,
        )

        results = retriever.retrieve("vitamin C")
        assert len(results) <= 2

    def test_get_context_with_citations(self, mock_embedder, mock_vector_store, sample_qa_data):
        """Should return context with numbered citations."""
        texts = [f"Câu hỏi: {qa['question']}\n\nTrả lời: {qa['answer']}" for qa in sample_qa_data]
        embeddings = mock_embedder.embed(texts)
        mock_vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=[{"question": qa["question"]} for qa in sample_qa_data],
            ids=[qa["id"] for qa in sample_qa_data],
        )

        retriever = Retriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            top_k=5,
        )

        result = retriever.get_context_with_citations("vitamin C", top_k=5)

        assert "context" in result
        assert "sources" in result
        assert isinstance(result["context"], str)

    def test_score_threshold_filtering(self, mock_embedder, mock_vector_store, sample_qa_data):
        """Should filter results by score threshold."""
        texts = [f"Câu hỏi: {qa['question']}\n\nTrả lời: {qa['answer']}" for qa in sample_qa_data]
        embeddings = mock_embedder.embed(texts)
        mock_vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=[{"question": qa["question"]} for qa in sample_qa_data],
            ids=[qa["id"] for qa in sample_qa_data],
        )

        retriever = Retriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            top_k=5,
            score_threshold=0.1,  # Very strict
            search_type="similarity_score_threshold",
        )

        results = retriever.retrieve("triệu chứng tiểu đường")
        # Should either be empty or have very few results
        for r in results:
            assert r["distance"] <= 0.1

    def test_hybrid_reranks_vector_candidates_with_keyword_signal(self, mock_embedder):
        """Hybrid mode should promote keyword-matching candidates."""
        class FakeVectorStore:
            def search(self, query_embedding, top_k=5):
                return [
                    {
                        "id": "dense-only",
                        "text": "general health article",
                        "metadata": {"question": "general"},
                        "distance": 0.1,
                    },
                    {
                        "id": "keyword-hit",
                        "text": "paracetamol qua lieu doc gan",
                        "metadata": {"question": "paracetamol"},
                        "distance": 0.4,
                    },
                ][:top_k]

        retriever = Retriever(
            vector_store=FakeVectorStore(),
            embedder=mock_embedder,
            top_k=1,
            search_type="hybrid",
            fetch_k=2,
            vector_weight=0.2,
            bm25_weight=0.8,
        )

        results = retriever.retrieve("paracetamol doc gan", top_k=1)

        assert results[0]["id"] == "keyword-hit"
        assert "hybrid_score" in results[0]

    def test_reranker_overrides_vector_order(self, mock_embedder, mock_vector_store):
        """When a reranker is supplied, its ordering should win over vector distance."""
        class FakeReranker:
            def rerank(self, query, documents, top_k=None):
                # Reverse the incoming order to prove the reranker's output is used.
                reversed_docs = list(reversed(documents))
                return reversed_docs[: top_k or len(reversed_docs)]

        class FakeVectorStore:
            def search(self, query_embedding, top_k=5):
                return [
                    {"id": "first", "text": "doc a", "metadata": {}, "distance": 0.1},
                    {"id": "second", "text": "doc b", "metadata": {}, "distance": 0.2},
                ][:top_k]

        retriever = Retriever(
            vector_store=FakeVectorStore(),
            embedder=mock_embedder,
            top_k=1,
            reranker=FakeReranker(),
        )

        results = retriever.retrieve("query", top_k=1)

        assert results[0]["id"] == "second"

    def test_no_reranker_keeps_original_order(self, mock_embedder):
        """Without a reranker, behavior is unchanged: top_k of the original order."""
        class FakeVectorStore:
            def search(self, query_embedding, top_k=5):
                return [
                    {"id": "first", "text": "doc a", "metadata": {}, "distance": 0.1},
                    {"id": "second", "text": "doc b", "metadata": {}, "distance": 0.2},
                ][:top_k]

        retriever = Retriever(
            vector_store=FakeVectorStore(),
            embedder=mock_embedder,
            top_k=1,
        )

        results = retriever.retrieve("query", top_k=1)

        assert results[0]["id"] == "first"
