"""Cross-encoder reranking for improved retrieval quality."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Reranker:
    """
    Rerank retrieved documents using a cross-encoder model.
    Cross-encoders provide more accurate relevance scoring than bi-encoders.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        top_k: int = 5,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        """
        Args:
            model_name: HuggingFace cross-encoder model
            device: "cuda", "cpu", or None (auto)
            top_k: Number of documents to return after reranking
            max_length: Max tokens per (query, document) pair
            batch_size: Pairs scored per forward pass (small keeps peak VRAM low)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self.top_k = top_k
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the cross-encoder model."""
        if self.model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, max_length=512, device=self.device)
            logger.info(f"Cross-encoder loaded: {self.model_name} on {self.device}")
        except ImportError:
            logger.warning("sentence-transformers not available, reranking disabled")
            self.model = None

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Rerank documents by cross-encoder relevance scores.

        Args:
            query: User query
            documents: List of retrieved documents with "text" key
            top_k: Override number of results to return

        Returns:
            Reranked documents with added "rerank_score" field
        """
        if not documents:
            return []

        if self.model is None:
            self.load()

        if self.model is None:
            return documents[:top_k or self.top_k]

        k = top_k or self.top_k

        # Prepare pairs for cross-encoder
        pairs = [(query, doc["text"]) for doc in documents]

        try:
            scores = self.model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False
            )
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original order")
            return documents[:k]

        # Add scores and sort
        scored_docs = []
        for i, doc in enumerate(documents):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(scores[i])
            doc_copy["original_position"] = i
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.debug(
            f"Reranked {len(documents)} documents, "
            f"top score: {scored_docs[0]['rerank_score']:.4f}"
        )

        return scored_docs[:k]


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
