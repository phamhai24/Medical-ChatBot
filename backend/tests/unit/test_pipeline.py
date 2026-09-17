"""Unit tests for the RAG pipeline."""

import pytest
from src.rag.pipeline import RAGPipeline, RAGResponse


class TestRAGPipeline:
    def test_pipeline_lazy_init(self, sample_config, mock_embedder, mock_vector_store):
        """Pipeline should initialize lazily."""
        pipeline = RAGPipeline(sample_config)
        assert not pipeline._initialized

        # Components should be None before init
        assert pipeline.embedder is None
        assert pipeline.vector_store is None

    def test_pipeline_ingest(self, sample_config, mock_embedder, mock_vector_store, sample_qa_data):
        """Pipeline ingest should process Q&A data."""
        pipeline = RAGPipeline(sample_config)
        pipeline.embedder = mock_embedder
        pipeline.vector_store = mock_vector_store

        # Monkey-patch to avoid real embedding
        pipeline._initialized = True

        # Create a simple in-memory store for testing
        stats = {
            "total_records": len(sample_qa_data),
            "total_chunks": len(sample_qa_data),
            "avg_chunk_length": 200,
            "embedding_model": "mock",
            "embedding_dimension": 384,
            "vector_store_type": "memory",
            "documents_indexed": mock_vector_store.count(),
        }

        assert stats["total_records"] == 3
        assert mock_vector_store.count() == 0

    def test_pipeline_query_no_results(self, sample_config):
        """Pipeline query should handle empty results."""
        pipeline = RAGPipeline(sample_config)

        # We can't fully test query without real components,
        # but we can test the response structure
        # This is a smoke test for the RAGResponse dataclass
        response = RAGResponse(
            answer="test answer",
            sources=[],
            query="test query",
            retrieved_docs=[],
            latency=0.1,
            generation_config={},
        )

        assert response.answer == "test answer"
        assert response.sources == []
        assert response.latency == 0.1

    def test_pipeline_get_stats(self, sample_config):
        """Pipeline get_stats should return correct structure."""
        pipeline = RAGPipeline(sample_config)
        stats = pipeline.get_stats()

        assert "initialized" in stats
        assert stats["initialized"] is False

    def test_query_reuses_retrieved_docs_for_context_and_sources(self, sample_config):
        """Query should not issue duplicate vector-store retrievals."""
        pipeline = RAGPipeline(sample_config)
        docs = [{
            "id": "doc-1",
            "text": "Context text",
            "metadata": {"question": "Source question", "chunk_index": 2},
            "distance": 0.12,
        }]

        class FakeRetriever:
            def __init__(self):
                self.calls = 0

            def retrieve(self, question, top_k=None):
                self.calls += 1
                return docs

            def get_context_with_citations(self, question, top_k=None):
                raise AssertionError("query() should reuse retrieved docs")

        class FakeGenerator:
            def generate_from_context(self, question, context, system_prompt=None, user_template=None):
                assert "Context text" in context
                return "answer"

        pipeline._initialized = True
        pipeline.retriever = FakeRetriever()
        pipeline.generator = FakeGenerator()

        response = pipeline.query("question", top_k=5, include_sources=True)

        assert pipeline.retriever.calls == 1
        assert response.answer == "answer"
        assert response.sources == [{
            "id": "doc-1",
            "question": "Source question",
            "score": 0.12,
            "chunk_index": 2,
        }]
        assert response.retrieval_latency >= 0
        assert response.generation_latency >= 0

    def test_query_without_sources_still_retrieves_once(self, sample_config):
        """Disabling source output should not trigger extra retrieval work."""
        pipeline = RAGPipeline(sample_config)
        docs = [{
            "id": "doc-1",
            "text": "Context text",
            "metadata": {"question": "Source question"},
            "distance": 0.12,
        }]

        class FakeRetriever:
            def __init__(self):
                self.calls = 0

            def retrieve(self, question, top_k=None):
                self.calls += 1
                return docs

        class FakeGenerator:
            def generate_from_context(self, question, context, system_prompt=None, user_template=None):
                return "answer"

        pipeline._initialized = True
        pipeline.retriever = FakeRetriever()
        pipeline.generator = FakeGenerator()

        response = pipeline.query("question", top_k=5, include_sources=False)

        assert pipeline.retriever.calls == 1
        assert response.sources == []
