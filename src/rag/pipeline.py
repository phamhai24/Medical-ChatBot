"""Main RAG Pipeline - Combines retrieval and generation"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.rag.chunker import TextChunker
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore, VectorStoreUnavailable
from src.rag.retriever import Retriever
from src.rag.generator import Generator

logger = logging.getLogger(__name__)


class _NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    retrieved_docs: List[Dict[str, Any]]
    latency: float
    generation_config: Dict[str, Any]
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0


class RAGPipeline:
    """
    Complete RAG pipeline: ingest data, retrieve context, generate response.

    Usage:
        # Ingest data
        pipeline = RAGPipeline(config)
        pipeline.ingest("data/processed/data.json")

        # Query
        response = pipeline.query("Triệu chứng của bệnh tiểu đường là gì?")
        print(response.answer)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary from rag_config.yaml
        """
        self.config = config
        self._initialized = False

        # Components
        self.chunker: Optional[TextChunker] = None
        self.embedder: Optional[Embedder] = None
        self.vector_store: Optional[VectorStore] = None
        self.retriever: Optional[Retriever] = None
        self.generator: Optional[Any] = None

        # Cached configs
        self._rag_config = config.get("rag", {})
        self._embedding_config = self._rag_config.get("embedding", {})
        self._vector_config = self._rag_config.get("vector_store", {})
        self._retrieval_config = self._rag_config.get("retrieval", {})
        self._generation_config = self._rag_config.get("generation", {})
        self._prompt_config = self._rag_config.get("prompt", {})

    def _lazy_init(self):
        """Initialize components lazily on first use."""
        if self._initialized:
            return

        logger.info("Initializing RAG pipeline components...")

        # Embedder
        self.embedder = Embedder(
            model_name=self._embedding_config.get("model_name"),
            device=self._embedding_config.get("device"),
            normalize=self._embedding_config.get("normalize_embeddings", True),
            batch_size=self._embedding_config.get("batch_size", 32),
            max_length=self._embedding_config.get("max_length", 512),
        )

        # Vector store
        self.vector_store = VectorStore.create(
            store_type=self._vector_config.get("type", "chroma"),
            persist_directory=self._vector_config.get("persist_directory"),
            collection_name=self._vector_config.get("collection_name", "medical_qa"),
            embedding_dimension=None,  # Will be set after loading embedder
        )

        # Optional cross-encoder reranker (final precision pass over candidates)
        reranker = None
        if self._retrieval_config.get("rerank_enabled", False):
            from src.rag.reranker import Reranker

            reranker = Reranker(
                model_name=self._retrieval_config.get("rerank_model", "BAAI/bge-reranker-v2-m3"),
                top_k=self._retrieval_config.get("top_k", 5),
            )

        # Retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=self._retrieval_config.get("top_k", 5),
            score_threshold=self._retrieval_config.get("score_threshold"),
            search_type=self._vector_config.get("search_type", "similarity"),
            fetch_k=self._retrieval_config.get("fetch_k", 20),
            vector_weight=self._retrieval_config.get("vector_weight", 0.6),
            bm25_weight=self._retrieval_config.get("bm25_weight", 0.4),
            reranker=reranker,
            rerank_fetch_k=self._retrieval_config.get("rerank_fetch_k", 20),
        )

        # Generator - local or API
        gen_mode = self._generation_config.get("mode", "local")
        if gen_mode == "api":
            from src.rag.api_generator import APIGenerator
            self.generator = APIGenerator(
                provider=self._generation_config.get("api_provider", "groq"),
                model=self._generation_config.get("model_name"),
                api_key=self._generation_config.get("api_key"),
                base_url=self._generation_config.get("api_base_url"),
                temperature=self._generation_config.get("temperature", 0.3),
                max_tokens=self._generation_config.get("max_new_tokens", 512),
                max_retries=self._generation_config.get("api_max_retries", 2),
                retry_backoff=self._generation_config.get("api_retry_backoff", 1.0),
                retry_max_backoff=self._generation_config.get(
                    "api_retry_max_backoff", 8.0
                ),
                circuit_breaker_threshold=self._generation_config.get(
                    "api_circuit_breaker_threshold", 3
                ),
                circuit_breaker_cooldown=self._generation_config.get(
                    "api_circuit_breaker_cooldown", 30.0
                ),
            )
        else:
            self.generator = Generator(
                model_name=self._generation_config.get("model_name"),
                max_new_tokens=self._generation_config.get("max_new_tokens", 512),
                temperature=self._generation_config.get("temperature", 0.3),
                top_p=self._generation_config.get("top_p", 0.9),
                top_k=self._generation_config.get("top_k", 50),
                do_sample=self._generation_config.get("do_sample", True),
                repetition_penalty=self._generation_config.get("repetition_penalty", 1.1),
                system_prompt=self._prompt_config.get("system"),
            )

        self._initialized = True
        logger.info(f"RAG pipeline initialized (generator_mode={gen_mode})")

    # ─── Data Ingestion ──────────────────────────────────────────

    def ingest(
        self,
        data_path: str,
        batch_size: int = 100,
        rebuild: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest medical Q&A data into the vector store.

        Args:
            data_path: Path to JSON data file
            batch_size: Batch size for embedding
            rebuild: Whether to rebuild the index from scratch
            show_progress: Show progress bar

        Returns:
            Ingestion statistics
        """
        self._lazy_init()

        import json
        from tqdm import tqdm

        logger.info(f"Starting ingestion from {data_path}")

        # Load data
        with open(data_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        total_records = len(data)
        logger.info(f"Loaded {total_records} Q&A records")

        # Initialize chunker
        self.chunker = TextChunker(
            chunk_size=self._rag_config.get("data", {}).get("chunk_size", 512),
            chunk_overlap=self._rag_config.get("data", {}).get("chunk_overlap", 64),
            min_chunk_length=self._rag_config.get("data", {}).get("min_chunk_length", 50),
        )

        # Chunk all documents
        if show_progress:
            logger.info("Chunking documents...")
        documents = self.chunker.chunk_qa_pairs(data, include_question=True)

        chunk_stats = self.chunker.get_chunk_stats(documents)
        logger.info(f"Created {chunk_stats['total_chunks']} chunks")

        self.embedder.load()
        dim = self.embedder.get_embedding_dimension()

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

        backup_path = None
        target_store = self.vector_store
        staging_store = None
        used_staging = False

        lock_context = (
            self.vector_store.write_lock()
            if hasattr(self.vector_store, "write_lock")
            else _NoopContext()
        )

        with lock_context:
            if rebuild and hasattr(self.vector_store, "create_staging_store"):
                logger.info(
                    "Rebuilding index in a staging ChromaDB directory before promotion..."
                )
                staging_store = self.vector_store.create_staging_store()
                target_store = staging_store
                used_staging = True
            elif rebuild:
                logger.info("Rebuilding index from scratch...")
                if hasattr(self.vector_store, "reset_storage"):
                    backup_path = self.vector_store.reset_storage(backup=True)
                else:
                    self.vector_store.clear()

            for i in tqdm(range(0, len(texts), batch_size), disable=not show_progress):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedder.embed(batch_texts)

                target_store.add_documents(
                    texts=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=metadatas[i:i + batch_size],
                    ids=ids[i:i + batch_size]
                )

            probe_embedding = self.embedder.embed_query(texts[0]) if texts else None

            expected_count = len(texts) if rebuild else None
            if hasattr(target_store, "validate_persisted"):
                documents_indexed = target_store.validate_persisted(
                    expected_count=expected_count,
                    probe_embedding=probe_embedding,
                )
            else:
                if hasattr(target_store, "close"):
                    target_store.close()
                target_store.load()
                documents_indexed = target_store.count()

            if used_staging:
                backup_path = self.vector_store.promote_from(staging_store, backup=True)
                self.vector_store.load()
                if hasattr(self.vector_store, "validate_persisted"):
                    promoted_count = self.vector_store.validate_persisted(
                        expected_count=len(texts),
                        probe_embedding=probe_embedding,
                    )
                    if promoted_count != documents_indexed:
                        raise VectorStoreUnavailable(
                            "Promoted ChromaDB store count changed from "
                            f"{documents_indexed} to {promoted_count}."
                        )
                    documents_indexed = promoted_count

        stats = {
            "total_records": total_records,
            "total_chunks": len(documents),
            "avg_chunk_length": chunk_stats["avg_length"],
            "embedding_model": self._embedding_config.get("model_name"),
            "embedding_dimension": dim,
            "vector_store_type": self._vector_config.get("type"),
            "collection": self._vector_config.get("collection_name"),
            "documents_indexed": documents_indexed,
        }
        if backup_path:
            stats["backup_path"] = backup_path

        logger.info(f"✅ Ingestion complete: {stats}")
        return stats

    # ─── Query / Inference ────────────────────────────────────────

    def _format_context_with_citations(
        self,
        docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format retrieved docs without issuing another vector-store query."""
        if not docs:
            return {"context": "", "sources": []}

        context_parts = []
        sources = []

        for i, doc in enumerate(docs):
            metadata = doc.get("metadata", {}) or {}
            question = metadata.get("question", "Unknown title")
            chunk_idx = metadata.get("chunk_index", 0)

            context_parts.append(f"[{i + 1}] {doc.get('text', '')}")
            sources.append({
                "id": doc.get("id"),
                "question": question,
                "score": doc.get("distance", 0),
                "chunk_index": chunk_idx,
            })

        return {
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
        }

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_sources: bool = True,
        return_raw_context: bool = False
    ) -> RAGResponse:
        """
        Query the RAG pipeline.

        Args:
            question: User question
            top_k: Number of documents to retrieve
            include_sources: Include source citations
            return_raw_context: Return raw context string instead of formatted

        Returns:
            RAGResponse object
        """
        self._lazy_init()
        start_time = time.perf_counter()

        # Retrieve
        retrieval_start = time.perf_counter()
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        retrieval_latency = time.perf_counter() - retrieval_start

        if not retrieved_docs:
            latency = time.perf_counter() - start_time
            return RAGResponse(
                answer="Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu để trả lời câu hỏi này. Vui lòng thử diễn đạt câu hỏi theo cách khác hoặc liên hệ chuyên gia y tế.",
                sources=[],
                query=question,
                retrieved_docs=[],
                latency=latency,
                generation_config={},
                retrieval_latency=retrieval_latency,
                generation_latency=0.0,
            )

        # Build context
        context_result = self._format_context_with_citations(retrieved_docs)
        if return_raw_context:
            context = retrieved_docs[0].get("text", "")
        else:
            context = context_result["context"]

        # Generate
        system_prompt = self._prompt_config.get("system", "")
        user_template = self._prompt_config.get("user_template", "")

        generation_start = time.perf_counter()
        answer = self.generator.generate_from_context(
            question=question,
            context=context,
            system_prompt=system_prompt,
            user_template=user_template,
        )
        generation_latency = time.perf_counter() - generation_start

        latency = time.perf_counter() - start_time

        # Build sources
        sources = context_result["sources"] if include_sources else []

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            retrieved_docs=retrieved_docs,
            latency=latency,
            generation_config={
                "model": self._generation_config.get("model_name"),
                "top_k": top_k or self._retrieval_config.get("top_k", 5),
                "temperature": self._generation_config.get("temperature", 0.3),
                "mode": self._generation_config.get("mode", "local"),
            },
            retrieval_latency=retrieval_latency,
            generation_latency=generation_latency,
        )

    def query_streaming(
        self,
        question: str,
        top_k: Optional[int] = None,
        callback=None
    ) -> str:
        """
        Query with streaming response.

        Args:
            question: User question
            top_k: Number of documents to retrieve
            callback: Function called with each text chunk

        Returns:
            Complete generated text
        """
        self._lazy_init()

        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

        if not retrieved_docs:
            return "Xin lỗi, tôi không tìm thấy thông tin phù hợp."

        context_result = self._format_context_with_citations(retrieved_docs)
        context = context_result["context"]

        system_prompt = self._prompt_config.get("system", "")
        user_template = self._prompt_config.get("user_template", "")

        prompt = user_template.format(question=question, context=context)

        return self.generator.generate_streaming(
            prompt=prompt,
            system_prompt=system_prompt,
            callback=callback,
        )

    # ─── Utility ───────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if not self._initialized:
            return {"initialized": False}

        vector_store_error = None
        metadata_count = None
        try:
            doc_count = self.vector_store.count() if self.vector_store else 0
        except Exception as e:
            doc_count = 0
            vector_store_error = str(e)
            if self.vector_store and hasattr(self.vector_store, "metadata_count"):
                metadata_count = self.vector_store.metadata_count()

        stats = {
            "initialized": True,
            "embedding_model": self._embedding_config.get("model_name"),
            "vector_store_type": self._vector_config.get("type"),
            "collection": self._vector_config.get("collection_name"),
            "document_count": doc_count,
            "generation_model": self._generation_config.get("model_name"),
            "generator_mode": self._generation_config.get("mode", "local"),
            "retrieval_top_k": self._retrieval_config.get("top_k", 5),
        }
        if vector_store_error:
            stats["vector_store_error"] = vector_store_error
        if metadata_count is not None:
            stats["metadata_document_count"] = metadata_count
        return stats

    def warm_up(self, probe_query: str = "kiểm tra sức khỏe hệ thống") -> Dict[str, Any]:
        """Eagerly initialize heavy components so the first user chat is faster."""
        self._lazy_init()

        timings: Dict[str, float] = {}
        start_total = time.perf_counter()

        start = time.perf_counter()
        self.embedder.load()
        timings["embedder_load_seconds"] = round(time.perf_counter() - start, 3)

        start = time.perf_counter()
        probe_embedding = self.embedder.embed_query(probe_query)
        timings["probe_embedding_seconds"] = round(time.perf_counter() - start, 3)

        start = time.perf_counter()
        doc_count = self.vector_store.count() if self.vector_store else 0
        timings["vector_count_seconds"] = round(time.perf_counter() - start, 3)

        retrieved_count = 0
        if doc_count > 0 and self.vector_store:
            start = time.perf_counter()
            retrieved_count = len(self.vector_store.search(probe_embedding, top_k=1))
            timings["probe_search_seconds"] = round(time.perf_counter() - start, 3)

        generator_mode = self._generation_config.get("mode", "local")
        if self.generator:
            start = time.perf_counter()
            if generator_mode == "api":
                load = getattr(self.generator, "load", None)
                if callable(load):
                    load()
                get_client = getattr(self.generator, "_get_client", None)
                if callable(get_client):
                    get_client()
            else:
                load = getattr(self.generator, "load", None)
                if callable(load):
                    load()
            timings["generator_warmup_seconds"] = round(time.perf_counter() - start, 3)

        timings["total_seconds"] = round(time.perf_counter() - start_total, 3)
        return {
            "document_count": doc_count,
            "probe_results": retrieved_count,
            "generator_mode": generator_mode,
            "timings": timings,
        }

    def check_health(self) -> Dict[str, Any]:
        """Check health of all pipeline components."""
        health = {"status": "healthy", "components": {}}

        try:
            if self.embedder:
                self.embedder.load()
                health["components"]["embedder"] = "ok"
        except Exception as e:
            health["components"]["embedder"] = f"error: {e}"
            health["status"] = "degraded"

        try:
            if self.vector_store:
                count = self.vector_store.count()
                health["components"]["vector_store"] = f"ok ({count} docs)"
        except Exception as e:
            health["components"]["vector_store"] = f"error: {e}"
            health["status"] = "unhealthy"

        return health
