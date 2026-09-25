"""Unit tests for config module."""

from src.core.config import Settings


class TestSettings:
    def test_default_values(self):
        """Settings should have sensible defaults."""
        settings = Settings()

        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.retrieval_top_k == 5
        assert settings.generator_temperature == 0.3

    def test_redis_url_without_password(self):
        """redis_url should be constructed correctly without password."""
        settings = Settings(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            redis_password=None,
        )
        assert settings.redis_url == "redis://localhost:6379/0"

    def test_redis_url_with_password(self):
        """redis_url should include password when set."""
        settings = Settings(
            redis_host="redis.example.com",
            redis_port=6380,
            redis_db=1,
            redis_password="secret",
        )
        assert "redis://:secret@" in settings.redis_url
        assert "redis.example.com" in settings.redis_url
        assert "/1" in settings.redis_url

    def test_is_cuda(self):
        """is_cuda should detect CUDA availability."""
        settings = Settings(embedding_device="cpu")
        assert settings.is_cuda is False

        settings = Settings(embedding_device="auto")
        # Just check it doesn't crash - actual value depends on hardware
        assert isinstance(settings.is_cuda, bool)
