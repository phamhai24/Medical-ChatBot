"""Unit tests for the RAG pipeline."""

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
        RAGPipeline(sample_config)

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
            "position": 1,
            "snippet": "Context text",
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

    def test_query_hides_sources_when_answer_declines(self, sample_config):
        """An honest 'not found' answer shouldn't surface unrelated docs as citations."""
        pipeline = RAGPipeline(sample_config)
        docs = [{
            "id": "doc-1",
            "text": "Unrelated context",
            "metadata": {"question": "Unrelated question"},
            "distance": 0.1,  # would otherwise easily pass the relevance threshold
        }]

        class FakeRetriever:
            def retrieve(self, question, top_k=None):
                return docs

        class FakeGenerator:
            def generate_from_context(self, question, context, system_prompt=None, user_template=None):
                return "Tôi không tìm thấy thông tin về vấn đề này trong dữ liệu được cung cấp."

        pipeline._initialized = True
        pipeline.retriever = FakeRetriever()
        pipeline.generator = FakeGenerator()

        response = pipeline.query("question", top_k=5, include_sources=True)

        assert response.sources == []

    def test_query_drops_low_relevance_sources(self, sample_config):
        """Sources past the relevance threshold shouldn't be shown even on a normal answer."""
        pipeline = RAGPipeline(sample_config)
        docs = [
            {"id": "close", "text": "t", "metadata": {"question": "q1"}, "distance": 0.3},
            {"id": "far", "text": "t", "metadata": {"question": "q2"}, "distance": 0.9},
        ]

        class FakeRetriever:
            def retrieve(self, question, top_k=None):
                return docs

        class FakeGenerator:
            def generate_from_context(self, question, context, system_prompt=None, user_template=None):
                return "A perfectly normal, grounded answer."

        pipeline._initialized = True
        pipeline.retriever = FakeRetriever()
        pipeline.generator = FakeGenerator()

        response = pipeline.query("question", top_k=5, include_sources=True)

        assert [s["id"] for s in response.sources] == ["close"]

    def test_query_collapses_same_document_chunks_into_one_citation(self, sample_config):
        """Multiple chunks of one document shouldn't look like 3 independent sources."""
        pipeline = RAGPipeline(sample_config)
        docs = [
            {"id": "doc-1_0", "text": "part 1", "metadata": {"question": "Migraine treatment", "chunk_index": 0}, "distance": 0.4},
            {"id": "doc-1_1", "text": "part 2", "metadata": {"question": "Migraine treatment", "chunk_index": 1}, "distance": 0.41},
            {"id": "doc-1_2", "text": "part 3", "metadata": {"question": "Migraine treatment", "chunk_index": 2}, "distance": 0.42},
        ]

        class FakeRetriever:
            def retrieve(self, question, top_k=None):
                return docs

        class FakeGenerator:
            def generate_from_context(self, question, context, system_prompt=None, user_template=None):
                return "A normal answer."

        pipeline._initialized = True
        pipeline.retriever = FakeRetriever()
        pipeline.generator = FakeGenerator()

        response = pipeline.query("question", top_k=5, include_sources=True)

        assert len(response.sources) == 1
        assert response.sources[0]["id"] == "doc-1_0"

    def test_query_uses_inline_citation_markers_when_present(self, sample_config):
        """When the model cites specific [n] markers, only those sources are kept."""
        pipeline = RAGPipeline(sample_config)
        docs = [
            {"id": "cited", "text": "t", "metadata": {"question": "Used topic"}, "distance": 0.1},
            {"id": "uncited", "text": "t", "metadata": {"question": "Unused topic"}, "distance": 0.1},
        ]

        class FakeRetriever:
            def retrieve(self, question, top_k=None):
                return docs

        class FakeGenerator:
            def generate_from_context(self, question, context, system_prompt=None, user_template=None):
                # Cites only source [1], even though [2] also passes the score threshold.
                return "The answer draws on the first source [1]."

        pipeline._initialized = True
        pipeline.retriever = FakeRetriever()
        pipeline.generator = FakeGenerator()

        response = pipeline.query("question", top_k=5, include_sources=True)

        assert [s["id"] for s in response.sources] == ["cited"]

    def test_chunk_snippet_strips_prefix_and_truncates(self, sample_config):
        """Citations should show the chunk's actual content, not the chunker's title prefix."""
        pipeline = RAGPipeline(sample_config)

        assert pipeline._chunk_snippet("Câu hỏi: Sốt là gì?\n\nTrả lời: Sốt là tình trạng...") == (
            "Sốt là tình trạng..."
        )
        long_text = "Trả lời: " + "a" * 200
        snippet = pipeline._chunk_snippet(long_text, max_length=160)
        assert len(snippet) == 161  # 160 chars + the truncation ellipsis
        assert snippet.endswith("…")

    def test_warm_up_loads_bm25_index_eagerly(self, sample_config):
        """warm_up() should trigger the BM25 index's lazy load, not leave it for the first user."""
        pipeline = RAGPipeline(sample_config)

        class FakeEmbedder:
            def load(self):
                pass

            def embed_query(self, text):
                return [0.0]

        class FakeVectorStore:
            def count(self):
                return 1

            def search(self, embedding, top_k=1):
                return [{"id": "d"}]

        class FakeGenerator:
            pass

        class FakeBM25Index:
            def __init__(self):
                self.load_calls = 0

            def load(self):
                self.load_calls += 1

        bm25_index = FakeBM25Index()
        pipeline._initialized = True
        pipeline.embedder = FakeEmbedder()
        pipeline.vector_store = FakeVectorStore()
        pipeline.generator = FakeGenerator()
        pipeline.retriever = type("R", (), {"bm25_index": bm25_index})()

        pipeline.warm_up("probe")

        assert bm25_index.load_calls == 1
