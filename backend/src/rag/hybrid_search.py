"""Hybrid search combining vector similarity with BM25 keyword search."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 (Best Matching 25) keyword-based retriever.
    Used in combination with vector search for hybrid retrieval.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        top_k: int = 10,
    ):
        """
        Args:
            k1: BM25 term frequency saturation parameter
            b: BM25 document length normalization parameter
            top_k: Number of results to return
        """
        self.k1 = k1
        self.b = b
        self.top_k = top_k
        self._index: Optional[dict] = None

    def build_index(self, documents: list[dict]):
        """Build BM25 index from documents."""
        from collections import Counter

        if not documents:
            self._index = None
            return

        self._documents = documents
        self._N = len(documents)

        # Tokenize all documents
        tokenized = [self._tokenize(doc.get("text", "")) for doc in documents]

        # Document lengths
        self._doc_lengths = [len(tokens) for tokens in tokenized]
        self._avgdl = sum(self._doc_lengths) / self._N if self._N > 0 else 0

        # Document frequencies
        df = Counter()
        for tokens in tokenized:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        self._df = dict(df)
        self._tokenized = tokenized

        logger.info(f"BM25 index built with {self._N} documents, {len(df)} unique terms")

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r"\w+", text, re.UNICODE)
        return tokens

    def get_scores(self, query: str) -> list[float]:
        """Calculate BM25 scores for all documents given a query."""
        import math
        from collections import Counter

        if not hasattr(self, "_documents") or self._documents is None:
            return [0.0]

        query_tokens = self._tokenize(query)
        scores = []

        for i, tokens in enumerate(self._tokenized):
            score = 0.0
            doc_freq = Counter(tokens)

            for token in query_tokens:
                if token not in self._df:
                    continue

                tf = doc_freq.get(token, 0)
                df_t = self._df[token]

                idf = math.log((self._N - df_t + 0.5) / (df_t + 0.5) + 1)
                tf_component = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * self._doc_lengths[i] / self._avgdl)
                )
                score += idf * tf_component

            scores.append(score)

        return scores

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Search BM25 index for top matching documents."""
        k = top_k or self.top_k

        if not hasattr(self, "_documents") or self._documents is None:
            return []

        scores = self.get_scores(query)

        # Pair documents with scores
        scored_docs = []
        for i, doc in enumerate(self._documents):
            doc_copy = doc.copy()
            doc_copy["bm25_score"] = scores[i]
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda x: x["bm25_score"], reverse=True)

        # Filter out zero scores
        scored_docs = [d for d in scored_docs if d["bm25_score"] > 0]

        return scored_docs[:k]


class HybridRetriever:
    """
    Combines vector search and BM25 search using Reciprocal Rank Fusion (RRF).
    RRF is simple, parameter-free, and highly effective.
    """

    def __init__(
        self,
        vector_retriever,
        bm25_retriever: BM25Retriever,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        """
        Args:
            vector_retriever: Vector-based retriever
            bm25_retriever: BM25 retriever
            vector_weight: Weight for vector search scores
            bm25_weight: Weight for BM25 scores
            rrf_k: RRF constant (higher = more weight to lower ranks)
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_top_k: int = 20,
    ) -> list[dict]:
        """
        Hybrid search combining vector and BM25 results.

        Args:
            query: User query
            top_k: Final number of results
            vector_top_k: Number of vector results to fetch

        Returns:
            Fused results ranked by RRF
        """
        # Vector search
        vector_results = self.vector_retriever.retrieve(query, top_k=vector_top_k)
        vector_scores = {
            r.get("id", i): self._convert_vector_score(r.get("distance", 1.0))
            for i, r in enumerate(vector_results)
        }

        # BM25 search
        bm25_results = self.bm25_retriever.search(query, top_k=vector_top_k)
        bm25_scores = {
            r.get("id", i): r.get("bm25_score", 0)
            for i, r in enumerate(bm25_results)
        }

        # Normalize scores
        vector_scores = self._normalize_scores(vector_scores)
        bm25_scores = self._normalize_scores(bm25_scores)

        # Combine all document IDs
        all_ids = set(list(vector_scores.keys())[:vector_top_k] + list(bm25_scores.keys())[:vector_top_k])

        # Reciprocal Rank Fusion
        fused_scores = {}
        for doc_id in all_ids:
            vs = vector_scores.get(doc_id, 0) * self.vector_weight
            bs = bm25_scores.get(doc_id, 0) * self.bm25_weight
            fused_scores[doc_id] = vs + bs

        # Build result list
        doc_map = {r.get("id"): r for r in vector_results}
        doc_map.update({r.get("id"): r for r in bm25_results})

        fused = []
        for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            if doc_id in doc_map:
                doc = doc_map[doc_id].copy()
                doc["hybrid_score"] = score
                doc["vector_score"] = vector_scores.get(doc_id, 0)
                doc["bm25_score"] = bm25_scores.get(doc_id, 0)
                fused.append(doc)

        return fused

    @staticmethod
    def _convert_vector_score(distance: float) -> float:
        """Convert distance (lower=better) to similarity (higher=better)."""
        return max(0, 1.0 - distance)

    @staticmethod
    def _normalize_scores(scores: dict) -> dict:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return scores
        vals = list(scores.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {k: 0.5 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}
