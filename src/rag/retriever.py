"""Retriever for RAG pipeline"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieve relevant documents from vector store.
    Supports multiple retrieval strategies.
    """

    def __init__(
        self,
        vector_store,
        embedder,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        search_type: str = "similarity",
        fetch_k: int = 20,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        """
        Args:
            vector_store: VectorStore instance
            embedder: Embedder instance
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            search_type: "similarity", "hybrid", or "similarity_score_threshold"
            fetch_k: Candidate pool size for hybrid retrieval
            vector_weight: Dense vector score weight for hybrid retrieval
            bm25_weight: Keyword score weight for hybrid retrieval
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.search_type = search_type
        self.fetch_k = fetch_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query
            top_k: Override number of results (optional)

        Returns:
            List of relevant documents with metadata
        """
        k = top_k or self.top_k

        # Embed the query
        query_embedding = self.embedder.embed_query(query)
        logger.debug(f"Query embedding shape: {len(query_embedding)}")

        # Search vector store
        if self.search_type == "similarity_score_threshold":
            # Fetch more, then filter
            raw_results = self.vector_store.search(query_embedding, top_k=k * 2)
            results = self._filter_by_threshold(raw_results)
            if len(results) >= k:
                results = results[:k]
            else:
                # Fall back to similarity search
                results = self.vector_store.search(query_embedding, top_k=k)
                results = self._filter_by_threshold(results)
        elif self.search_type == "hybrid":
            fetch_k = max(k, self.fetch_k)
            raw_results = self.vector_store.search(query_embedding, top_k=fetch_k)
            results = self._hybrid_rerank(query, raw_results, top_k=k)
        else:
            results = self.vector_store.search(query_embedding, top_k=k)

        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results

    def _hybrid_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Rerank vector candidates with a lightweight BM25 keyword signal."""
        if not candidates:
            return []

        try:
            from src.rag.hybrid_search import BM25Retriever

            bm25 = BM25Retriever(top_k=len(candidates))
            bm25.build_index(candidates)
            bm25_results = bm25.search(query, top_k=len(candidates))
        except Exception as exc:
            logger.warning("Hybrid BM25 rerank failed, using vector order: %s", exc)
            return candidates[:top_k]

        bm25_scores = {
            result.get("id") if result.get("id") is not None else index: result.get("bm25_score", 0.0)
            for index, result in enumerate(bm25_results)
        }

        vector_scores = {
            doc.get("id") if doc.get("id") is not None else index: max(0.0, 1.0 - float(doc.get("distance", 1.0)))
            for index, doc in enumerate(candidates)
        }

        bm25_scores = self._normalize_scores(bm25_scores)
        vector_scores = self._normalize_scores(vector_scores)

        reranked = []
        for index, doc in enumerate(candidates):
            doc_id = doc.get("id") if doc.get("id") is not None else index
            vector_score = vector_scores.get(doc_id, 0.0)
            bm25_score = bm25_scores.get(doc_id, 0.0)
            hybrid_score = (
                vector_score * self.vector_weight
                + bm25_score * self.bm25_weight
            )

            doc_copy = doc.copy()
            doc_copy["vector_score"] = vector_score
            doc_copy["bm25_score"] = bm25_score
            doc_copy["hybrid_score"] = hybrid_score
            reranked.append(doc_copy)

        reranked.sort(key=lambda doc: doc.get("hybrid_score", 0.0), reverse=True)
        return self._deduplicate_results(reranked)[:top_k]

    @staticmethod
    def _deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prefer diverse source questions, then fill with remaining chunks."""
        primary = []
        overflow = []
        seen_questions = set()
        seen_chunks = set()

        for doc in results:
            metadata = doc.get("metadata", {}) or {}
            question = metadata.get("question")
            chunk_index = metadata.get("chunk_index")
            text = doc.get("text", "")
            chunk_key = (
                question or doc.get("id") or text[:160],
                chunk_index if chunk_index is not None else text[:160],
            )
            if chunk_key in seen_chunks:
                continue
            seen_chunks.add(chunk_key)

            question_key = question or chunk_key
            if question_key in seen_questions:
                overflow.append(doc)
            else:
                seen_questions.add(question_key)
                primary.append(doc)

        return primary + overflow

    @staticmethod
    def _normalize_scores(scores: Dict[Any, float]) -> Dict[Any, float]:
        """Normalize score values to [0, 1]."""
        if not scores:
            return scores

        values = list(scores.values())
        min_value = min(values)
        max_value = max(values)
        if max_value == min_value:
            return {key: 0.5 for key in scores}

        return {
            key: (value - min_value) / (max_value - min_value)
            for key, value in scores.items()
        }

    def _filter_by_threshold(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter results by similarity score threshold."""
        if self.score_threshold is None:
            return results

        return [
            r for r in results
            if r.get("distance", 1.0) <= self.score_threshold
        ]

    def retrieve_with_expansion(
        self,
        query: str,
        top_k: Optional[int] = None,
        n_expansions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with query expansion using synonyms.
        Simple implementation: combines original query with top terms.
        """
        # Simple query expansion: just retrieve more
        k = (top_k or self.top_k) * n_expansions
        return self.retrieve(query, top_k=k)

    def get_relevant_context(
        self,
        query: str,
        max_context_length: int = 4000
    ) -> str:
        """
        Get a formatted context string from retrieved documents.

        Args:
            query: User query
            max_context_length: Maximum characters in context

        Returns:
            Formatted context string
        """
        docs = self.retrieve(query)

        if not docs:
            return ""

        context_parts = []
        current_length = 0

        for doc in docs:
            doc_text = doc["text"]
            if current_length + len(doc_text) + 2 > max_context_length:
                break
            context_parts.append(doc_text)
            current_length += len(doc_text) + 2

        context = "\n\n---\n\n".join(context_parts)
        return context

    def get_context_with_citations(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get context with source citations.

        Returns:
            {"context": str, "sources": List[Dict]}
        """
        docs = self.retrieve(query, top_k=top_k)

        if not docs:
            return {"context": "", "sources": []}

        context_parts = []
        sources = []

        for i, doc in enumerate(docs):
            question = doc.get("metadata", {}).get("question", "Không có tiêu đề")
            chunk_idx = doc.get("metadata", {}).get("chunk_index", 0)

            source_label = f"[{i + 1}] Câu hỏi: {question}"
            if doc.get("metadata", {}).get("total_chunks", 1) > 1:
                source_label += f" (phần {chunk_idx + 1})"

            context_parts.append(f"[{i + 1}] {doc['text']}")
            sources.append({
                "id": doc.get("id"),
                "question": question,
                "score": doc.get("distance", 0),
                "chunk_index": chunk_idx,
            })

        return {
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources
        }
