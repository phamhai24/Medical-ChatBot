"""Embedding model wrapper for text vectorization"""

import logging
import torch
from typing import List, Union, Optional

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wrapper for sentence embedding models.
    Supports HuggingFace sentence-transformers and custom models.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
        max_length: int = 512
    ):
        """
        Args:
            model_name: HuggingFace model name for embeddings
            device: "cuda", "cpu", or None (auto-detect)
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for embedding generation
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Embedder initialized (device: {self.device})")

    def load(self):
        """Load the embedding model."""
        if self.model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )
        self.model.max_seq_length = self.max_length
        logger.info(f"Embedding model loaded on {self.device}")

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: Single text or list of texts

        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self.load()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )

        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query (optimized for retrieval).

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        return self.embed(query)[0]

    def get_embedding_dimension(self) -> int:
        """Get the embedding vector dimension."""
        if self.model is None:
            self.load()

        dummy_embedding = self.model.encode("test")
        return dummy_embedding.shape[-1]

    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "embedding_dim": self.get_embedding_dimension() if self.model else None,
        }
