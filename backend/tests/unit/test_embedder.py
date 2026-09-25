"""Unit tests for the Embedder."""



class TestEmbedder:
    def test_embed_single_text(self, mock_embedder):
        """Single text should return a single embedding."""
        result = mock_embedder.embed("Triệu chứng bệnh tiểu đường")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 384

    def test_embed_multiple_texts(self, mock_embedder):
        """Multiple texts should return multiple embeddings."""
        texts = [
            "Triệu chứng bệnh tiểu đường",
            "Cách phòng ngừa cao huyết áp",
        ]
        result = mock_embedder.embed(texts)
        assert len(result) == 2
        for emb in result:
            assert len(emb) == 384

    def test_embed_query(self, mock_embedder):
        """embed_query should return single embedding vector."""
        result = mock_embedder.embed_query("Triệu chứng bệnh tiểu đường")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_embedding_dimension(self, mock_embedder):
        """get_embedding_dimension should return correct dimension."""
        dim = mock_embedder.get_embedding_dimension()
        assert dim == 384

    def test_model_info(self, mock_embedder):
        """get_model_info should return model metadata."""
        info = mock_embedder.get_model_info()
        assert "model_name" in info
        assert "device" in info
        assert "loaded" in info
