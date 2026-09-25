"""Whole-corpus BM25 keyword index, backed by the bm25s library.

Unlike Retriever._hybrid_rerank's candidate-only BM25 (which only re-scores
whatever the vector search already picked as its top-`fetch_k` candidates —
see the class docstring below for why that misses exact-term matches vector
search ranks low), this indexes every chunk in the corpus so a query can
surface a document purely on keyword match, independent of what the vector
search returned. A naive per-query brute-force scan over ~700k documents was
measured at ~10s/query (see reports/) — unusable for chat; bm25s's sparse-
matrix implementation measured at ~11ms/query for the same corpus.

Build with `scripts/build_bm25_index.py`; this class only loads and queries
the result. If the index hasn't been built yet, RAGPipeline falls back to
the existing candidate-only behavior — this is purely additive.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PersistedBM25Index:
    """Loads a bm25s index built by scripts/build_bm25_index.py and serves queries."""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self._retriever = None

    @property
    def is_available(self) -> bool:
        return Path(self.index_dir).exists()

    def load(self) -> bool:
        """Load the persisted index. Returns False (not an error) if none exists yet."""
        if self._retriever is not None:
            return True
        if not self.is_available:
            logger.info(f"No BM25 index found at {self.index_dir}; skipping whole-corpus BM25")
            return False

        import bm25s

        self._retriever = bm25s.BM25.load(
            self.index_dir, load_corpus=True, show_progress=False
        )
        logger.info(f"Loaded whole-corpus BM25 index from {self.index_dir}")
        return True

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Search the whole corpus. Returns docs shaped like VectorStore.search()'s output."""
        if self._retriever is None and not self.load():
            return []

        import bm25s

        query_tokens = bm25s.tokenize([query], stopwords=None, show_progress=False)
        k = min(top_k, len(self._retriever.corpus))
        if k == 0:
            return []
        results, scores = self._retriever.retrieve(query_tokens, k=k, show_progress=False)

        docs = []
        for entry, score in zip(results[0], scores[0]):
            if float(score) <= 0:
                continue
            docs.append({
                "id": entry.get("id"),
                "text": entry.get("text", ""),
                "metadata": entry.get("metadata", {}),
                "bm25_score": float(score),
            })
        return docs
