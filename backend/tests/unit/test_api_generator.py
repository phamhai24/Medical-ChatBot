"""Unit tests for API-based generator retry behavior."""

import httpx

from src.rag.api_generator import APICircuitBreakerOpen, APIGenerator


class FakeResponse:
    def __init__(self, status_code=200, data=None, headers=None):
        self.status_code = status_code
        self._data = data or {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 3},
        }
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.test/chat/completions")
            response = httpx.Response(
                self.status_code,
                request=request,
                headers=self.headers,
            )
            raise httpx.HTTPStatusError(
                "temporary error",
                request=request,
                response=response,
            )

    def json(self):
        return self._data


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, path, json):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_generate_retries_rate_limit(monkeypatch):
    """429 responses should be retried before succeeding."""
    generator = APIGenerator(
        provider="groq",
        model="test-model",
        api_key="test-key",
        max_retries=2,
        retry_backoff=0,
    )
    client = FakeClient([
        FakeResponse(status_code=429),
        FakeResponse(data={"choices": [{"message": {"content": "retry ok"}}]}),
    ])
    generator._client = client

    monkeypatch.setattr("src.rag.api_generator.time.sleep", lambda delay: None)

    assert generator.generate("hello") == "retry ok"
    assert client.calls == 2


def test_generate_does_not_retry_bad_request(monkeypatch):
    """Non-transient HTTP errors should fail immediately."""
    generator = APIGenerator(
        provider="groq",
        model="test-model",
        api_key="test-key",
        max_retries=2,
        retry_backoff=0,
    )
    client = FakeClient([FakeResponse(status_code=400)])
    generator._client = client

    monkeypatch.setattr("src.rag.api_generator.time.sleep", lambda delay: None)

    try:
        generator.generate("hello")
    except httpx.HTTPStatusError:
        pass
    else:
        raise AssertionError("Expected HTTPStatusError")

    assert client.calls == 1


def test_generate_opens_circuit_after_consecutive_retryable_failures(monkeypatch):
    """Repeated retryable failures should open the circuit breaker."""
    generator = APIGenerator(
        provider="groq",
        model="test-model",
        api_key="test-key",
        max_retries=0,
        retry_backoff=0,
        circuit_breaker_threshold=2,
        circuit_breaker_cooldown=30,
    )
    client = FakeClient([
        FakeResponse(status_code=429),
        FakeResponse(status_code=429),
    ])
    generator._client = client

    monkeypatch.setattr("src.rag.api_generator.time.sleep", lambda delay: None)

    for _ in range(2):
        try:
            generator.generate("hello")
        except httpx.HTTPStatusError:
            pass
        else:
            raise AssertionError("Expected HTTPStatusError")

    try:
        generator.generate("hello")
    except APICircuitBreakerOpen:
        pass
    else:
        raise AssertionError("Expected APICircuitBreakerOpen")

    assert client.calls == 2
