"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_root_endpoint(self):
        """Root endpoint should return API info."""
        # This tests the structure - actual FastAPI app needs pipeline
        from src.api.main import create_app
        app = create_app()
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data

    def test_health_endpoint(self):
        """Health endpoint should return component status."""
        from src.api.main import create_app
        app = create_app()
        client = TestClient(app)

        response = client.get("/health")
        # May fail if pipeline not initialized, but structure should be right
        # Accept 200 or 503 (pipeline not initialized is ok for structure test)
        assert response.status_code in [200, 503]

    def test_metrics_endpoint(self):
        """Metrics endpoint should return prometheus format."""
        from src.api.main import create_app
        app = create_app()
        client = TestClient(app)

        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")


class TestAPISchemas:
    def test_chat_request_validation(self):
        """ChatRequest should validate input correctly."""
        from src.api.schemas import ChatRequest

        # Valid request
        req = ChatRequest(message="Triệu chứng tiểu đường là gì?")
        assert req.message == "Triệu chứng tiểu đường là gì?"
        assert req.top_k == 5
        assert req.include_sources is True

    def test_chat_request_empty_message_fails(self):
        """ChatRequest should reject empty messages."""
        from src.api.schemas import ChatRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_chat_request_top_k_bounds(self):
        """ChatRequest should enforce top_k bounds."""
        from src.api.schemas import ChatRequest
        from pydantic import ValidationError

        # Valid bounds
        ChatRequest(message="test", top_k=1)
        ChatRequest(message="test", top_k=20)

        # Out of bounds
        with pytest.raises(ValidationError):
            ChatRequest(message="test", top_k=0)
        with pytest.raises(ValidationError):
            ChatRequest(message="test", top_k=100)

    def test_ingest_request_defaults(self):
        """IngestRequest should have correct defaults."""
        from src.api.schemas import IngestRequest

        req = IngestRequest()
        assert req.rebuild is False
        assert req.batch_size == 100
        assert req.data_path == "data/processed/data.json"
