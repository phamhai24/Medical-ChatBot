"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture
def sample_qa_data():
    return [
        {
            "id": "test_001",
            "question": "Triệu chứng bệnh tiểu đường là gì?",
            "answer": "Bệnh tiểu đường type 2 có các triệu chứng thường gặp bao gồm: khát nhiều nước, đi tiểu thường xuyên, mệt mỏi, nhìn mờ, vết thương lâu lành. Đường huyết tăng cao là dấu hiệu chính của bệnh.",
            "category": "nội tiết",
        },
        {
            "id": "test_002",
            "question": "Cách phòng ngừa bệnh cao huyết áp?",
            "answer": "Phòng ngừa bệnh cao huyết áp bằng cách: ăn giảm muối, tăng cường rau quả, tập thể dục đều đặn 30 phút mỗi ngày, duy trì cân nặng hợp lý, hạn chế rượu bia, không hút thuốc.",
            "category": "tim mạch",
        },
        {
            "id": "test_003",
            "question": "Vitamin C có lợi ích gì?",
            "answer": "Vitamin C có nhiều lợi ích: tăng cường miễn dịch, chống oxy hóa, giúp hấp thu sắt từ thực phẩm, duy trì sức khỏe da và mạch máu, giảm nguy cơ bệnh tim mạch.",
            "category": "dinh dưỡng",
        },
    ]


@pytest.fixture
def sample_config():
    return {
        "rag": {
            "data": {
                "chunk_size": 128,
                "chunk_overlap": 32,
                "min_chunk_length": 10,
            },
            "embedding": {
                "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "device": "cpu",
                "normalize_embeddings": True,
                "batch_size": 8,
                "max_length": 128,
            },
            "vector_store": {
                "type": "memory",
                "persist_directory": None,
                "collection_name": "test_collection",
                "search_type": "similarity",
                "fetch_k": 10,
                "lambda_mult": 0.5,
            },
            "retrieval": {
                "top_k": 5,
                "score_threshold": None,
            },
            "generation": {
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_new_tokens": 128,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": False,
                "repetition_penalty": 1.1,
                "mode": "local",
            },
            "prompt": {
                "system": "Bạn là trợ lý y tế. Trả lời ngắn gọn.",
                "user_template": "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        }
    }


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns deterministic fake embeddings."""
    import numpy as np

    class MockEmbedder:
        def __init__(self):
            self.model = "mock"

        def load(self):
            pass

        def embed(self, texts, show_progress=None):
            # Return deterministic fake embeddings
            import hashlib
            # Same contract as the real Embedder.embed: a single string is one
            # text, not an iterable of characters.
            if isinstance(texts, str):
                texts = [texts]
            embeddings = []
            for text in texts:
                # Create fake embedding based on text hash
                vec = np.zeros(384)
                seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
                np.random.seed(seed % (2**31))
                vec[:10] = np.random.randn(10)
                vec = vec / np.linalg.norm(vec)
                embeddings.append(vec.tolist())
            return embeddings

        def embed_query(self, query):
            return self.embed([query])[0]

        def get_embedding_dimension(self):
            return 384

        def get_model_info(self):
            return {"model_name": "mock", "device": "cpu", "loaded": True}

    return MockEmbedder()


@pytest.fixture
def mock_vector_store():
    """Mock vector store backed by dict."""
    class MockVectorStore:
        def __init__(self):
            self.docs = []
            self.embeddings = []
            self.metadatas = []
            self.ids = []
            self._id_counter = 0

        def load(self):
            pass

        def add_documents(self, texts, embeddings, metadatas=None, ids=None):
            for i, text in enumerate(texts):
                self.docs.append(text)
                self.embeddings.append(embeddings[i])
                self.metadatas.append(metadatas[i] if metadatas else {})
                self.ids.append(ids[i] if ids else f"doc_{self._id_counter}")
                self._id_counter += 1

        def search(self, query_embedding, top_k=5):
            import numpy as np

            if not self.docs:
                return []

            query = np.array(query_embedding)
            results = []

            for i, emb in enumerate(self.embeddings):
                emb_arr = np.array(emb)
                dist = 1 - float(np.dot(query, emb_arr))
                results.append({
                    "text": self.docs[i],
                    "metadata": self.metadatas[i],
                    "distance": dist,
                    "id": self.ids[i],
                })

            results.sort(key=lambda x: x["distance"])
            return results[:top_k]

        def count(self):
            return len(self.docs)

        def clear(self):
            self.docs = []
            self.embeddings = []
            self.metadatas = []
            self.ids = []
            self._id_counter = 0

    return MockVectorStore()
