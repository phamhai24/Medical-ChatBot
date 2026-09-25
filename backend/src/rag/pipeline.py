"""Main RAG Pipeline - Combines retrieval and generation"""

import logging
import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.rag.bm25_index import PersistedBM25Index
from src.rag.chunker import TextChunker
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore, VectorStoreUnavailable
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.utils.answer_signals import SOURCE_RELEVANCE_THRESHOLD, looks_like_decline

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
                max_length=self._retrieval_config.get("rerank_max_length", 512),
                batch_size=self._retrieval_config.get("rerank_batch_size", 16),
            )

        # Optional whole-corpus BM25 index (see src/rag/bm25_index.py for why this
        # is not the same as the candidate-only BM25 rerank below) — additive:
        # silently absent until `python scripts/build_bm25_index.py` has been run.
        bm25_index = None
        bm25_index_path = self._retrieval_config.get("bm25_index_path", "data/bm25_index")
        candidate_bm25_index = PersistedBM25Index(bm25_index_path)
        if candidate_bm25_index.is_available:
            bm25_index = candidate_bm25_index

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
            bm25_index=bm25_index,
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

            # ONE progress line for the whole run, rewritten in place (unit =
            # iteration = one batch). Kept short (no wide bar, fixed ncols) so it
            # never wraps to a new terminal line, and refreshed at most once per
            # second. The embedder's own per-call bar is suppressed, otherwise
            # every batch would print an extra "Batches" line.
            progress = tqdm(
                total=(len(texts) + batch_size - 1) // batch_size,
                desc="Ingest",
                unit="it",
                disable=not show_progress,
                ncols=70,
                mininterval=1.0,
                bar_format="{desc} {n_fmt}/{total_fmt} it [{elapsed}<{remaining}, {rate_fmt}]",
            )
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedder.embed(batch_texts, show_progress=False)

                target_store.add_documents(
                    texts=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=metadatas[i:i + batch_size],
                    ids=ids[i:i + batch_size]
                )
                progress.update(1)
            progress.close()

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
        seen_questions = set()

        for i, doc in enumerate(docs):
            metadata = doc.get("metadata", {}) or {}
            question = metadata.get("question", "Unknown title")
            chunk_idx = metadata.get("chunk_index", 0)

            context_parts.append(f"[{i + 1}] {doc.get('text', '')}")

            # Multiple chunks of the same source document shouldn't each count as
            # a separate "citation" — that inflates one weak match into what
            # looks like several independent supporting sources (e.g. a document
            # split into 3 chunks all landing in the top-3 when nothing else
            # matched). Keep only the first (best-ranked) chunk per document.
            if question in seen_questions:
                continue
            seen_questions.add(question)
            sources.append({
                "id": doc.get("id"),
                "question": question,
                "score": doc.get("distance", 0),
                "chunk_index": chunk_idx,
                # The [n] label this source was shown as in the LLM's context —
                # the frontend displays sources under this same number so an
                # inline "...[2][3]." in the answer matches the sources panel
                # instead of the panel renumbering from 1.
                "position": i + 1,
                # Source records in this corpus are often long articles (one
                # record can be 100+ chunks — see reports/ for the "Cao huyết
                # áp - hồi chuông cảnh báo" example, 66,948 chars / 149 chunks).
                # The parent "question" title describes the whole article, not
                # necessarily this specific chunk, so show an excerpt of the
                # actual retrieved text alongside it for an honest citation.
                "snippet": self._chunk_snippet(doc.get("text", "")),
            })

        return {
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
            # Maps each [n] label shown to the LLM to the question it belongs
            # to, so a citation like "[2]" can be resolved back to a source
            # even after _extract_cited_questions collapses duplicate chunks.
            "position_to_question": {
                i + 1: (doc.get("metadata", {}) or {}).get("question", "Unknown title")
                for i, doc in enumerate(docs)
            },
        }

    @staticmethod
    def _extract_cited_questions(answer: str, position_to_question: Dict[int, str]) -> set:
        """Parse [n] markers the model actually wrote and resolve them to source questions."""
        positions = {int(n) for n in re.findall(r"\[(\d+)\]", answer or "")}
        return {position_to_question[p] for p in positions if p in position_to_question}

    @staticmethod
    def _chunk_snippet(chunk_text: str, max_length: int = 160) -> str:
        """Excerpt of the actual chunk content, without the "Câu hỏi: ...\\n\\nTrả lời: " prefix
        the chunker prepends (see TextChunker.chunk_qa_pairs) — that prefix just repeats the
        source's title, which is already shown separately."""
        text = chunk_text or ""
        marker = "Trả lời:"
        idx = text.find(marker)
        if idx != -1:
            text = text[idx + len(marker):]
        text = text.strip()
        if len(text) > max_length:
            text = text[:max_length].rstrip() + "…"
        return text

    @staticmethod
    def _filter_sources_for_display(
        answer: str,
        sources: List[Dict[str, Any]],
        cited_questions: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Drop sources that don't plausibly back the answer.

        Three checks, in order: if the answer itself reads as "I didn't find
        this", none of the retrieved docs actually grounded it, so hide all of
        them. If the model wrote [n] citation markers (per the system prompt),
        trust them and keep only the sources it actually pointed to — the most
        precise signal available. Otherwise (no markers — not every generation
        follows the instruction), fall back to dropping individual sources
        whose vector-store distance is past SOURCE_RELEVANCE_THRESHOLD, a
        coarser secondary net (this threshold alone can't perfectly separate
        good from bad matches; see answer_signals.py for why).
        """
        if not sources:
            return sources
        if looks_like_decline(answer):
            return []
        if cited_questions:
            return [s for s in sources if s.get("question") in cited_questions]
        return [s for s in sources if s.get("score", 0) <= SOURCE_RELEVANCE_THRESHOLD]

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_sources: bool = True,
        return_raw_context: bool = False,
        generation_question: Optional[str] = None,
    ) -> RAGResponse:
        """
        Query the RAG pipeline.

        Args:
            question: User question (used for retrieval)
            top_k: Number of documents to retrieve
            include_sources: Include source citations
            return_raw_context: Return raw context string instead of formatted
            generation_question: What the answer model sees, if different from
                `question` (e.g. with the conversation topic noted; see
                src/rag/query_condenser.py). Defaults to `question`.

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
            question=generation_question or question,
            context=context,
            system_prompt=system_prompt,
            user_template=user_template,
        )
        generation_latency = time.perf_counter() - generation_start

        latency = time.perf_counter() - start_time

        # Build sources: only surface citations that plausibly back the answer.
        # An honest "not found" answer means the retrieved docs weren't relevant,
        # even though they were fed to the LLM as candidate context — showing
        # them as "sources" would misrepresent unrelated content as citations.
        sources = context_result["sources"] if include_sources else []
        cited_questions = self._extract_cited_questions(answer, context_result["position_to_question"])
        sources = self._filter_sources_for_display(answer, sources, cited_questions)

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

        # Step 1: Embedder
        embedder_name = getattr(self.embedder, "model_name", "BAAI/bge-m3")
        embedder_dev = getattr(self.embedder, "device", "unknown")
        logger.info(f"⏳ [1/6] Loading Embedding model '{embedder_name}' on device '{embedder_dev}'...")
        start = time.perf_counter()
        self.embedder.load()
        timings["embedder_load_seconds"] = round(time.perf_counter() - start, 3)
        logger.info(f"✅ [1/6] Embedding model ready in {timings['embedder_load_seconds']}s")

        # Step 2: Probe Embedding
        logger.info(f"⏳ [2/6] Running probe embedding query ('{probe_query[:30]}...')...")
        start = time.perf_counter()
        probe_embedding = self.embedder.embed_query(probe_query)
        timings["probe_embedding_seconds"] = round(time.perf_counter() - start, 3)
        logger.info(f"✅ [2/6] Probe embedding completed in {timings['probe_embedding_seconds']}s (dim: {len(probe_embedding)})")

        # Step 3: Vector Store Count & Probe Search
        logger.info("⏳ [3/6] Verifying Vector Store (counting indexed documents)...")
        start = time.perf_counter()
        doc_count = self.vector_store.count() if self.vector_store else 0
        timings["vector_count_seconds"] = round(time.perf_counter() - start, 3)

        retrieved_count = 0
        if doc_count > 0 and self.vector_store:
            start = time.perf_counter()
            retrieved_count = len(self.vector_store.search(probe_embedding, top_k=1))
            timings["probe_search_seconds"] = round(time.perf_counter() - start, 3)
        logger.info(f"✅ [3/6] Vector store verified: {doc_count:,} documents indexed in {timings['vector_count_seconds']}s")

        # Step 4: BM25 Index
        bm25_index = getattr(self.retriever, "bm25_index", None) if self.retriever else None
        if bm25_index is not None:
            logger.info("⏳ [4/6] Loading BM25 search index...")
            start = time.perf_counter()
            bm25_index.load()
            timings["bm25_index_load_seconds"] = round(time.perf_counter() - start, 3)
            logger.info(f"✅ [4/6] BM25 index loaded in {timings['bm25_index_load_seconds']}s")
        else:
            logger.info("ℹ️ [4/6] BM25 index not configured, skipping")

        # Step 5: Cross-Encoder Reranker
        reranker = getattr(self.retriever, "reranker", None) if self.retriever else None
        if reranker is not None:
            reranker_model = getattr(reranker, "model_name", "cross-encoder")
            reranker_dev = getattr(reranker, "device", "unknown")
            logger.info(f"⏳ [5/6] Loading Reranker model '{reranker_model}' on '{reranker_dev}'...")
            start = time.perf_counter()
            reranker.load()
            timings["reranker_load_seconds"] = round(time.perf_counter() - start, 3)

            # First rerank on CUDA pays a one-off kernel/cuBLAS init (~4s); absorb
            # it here with a throwaway pair instead of on the first user chat.
            start = time.perf_counter()
            reranker.rerank(probe_query, [{"text": probe_query}], top_k=1)
            timings["reranker_probe_seconds"] = round(time.perf_counter() - start, 3)
            logger.info(f"✅ [5/6] Reranker model ready in {timings['reranker_load_seconds']}s")
        else:
            logger.info("ℹ️ [5/6] Reranker not enabled, skipping")

        # Step 6: LLM Generator
        generator_mode = self._generation_config.get("mode", "local")
        gen_model = self._generation_config.get("model_name", "unknown")
        logger.info(f"⏳ [6/6] Initializing Generator ({generator_mode} mode - '{gen_model}')...")
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
            logger.info(f"✅ [6/6] Generator ready in {timings['generator_warmup_seconds']}s")

        timings["total_seconds"] = round(time.perf_counter() - start_total, 3)
        logger.info(f"🎉 Warm-up complete in {timings['total_seconds']}s! All components ready.")
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
