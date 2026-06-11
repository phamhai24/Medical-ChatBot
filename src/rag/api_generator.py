"""API-based LLM generator supporting Groq, OpenAI, and other compatible APIs."""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class APICircuitBreakerOpen(RuntimeError):
    """Raised when the upstream API is temporarily unavailable."""


class APIGenerator:
    """
    LLM generator using external API endpoints (Groq, OpenAI, etc.).
    Falls back gracefully when API key is not available.
    """

    def __init__(
        self,
        provider: str = "groq",
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        timeout: int = 60,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        retry_max_backoff: float = 8.0,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown: float = 30.0,
    ):
        """
        Args:
            provider: "groq", "openai", "anthropic", or "custom"
            model: Model name to use
            api_key: API key for the provider
            base_url: Custom base URL (overrides provider default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Number of retries for transient API failures
            retry_backoff: Initial retry delay in seconds
            retry_max_backoff: Maximum retry delay in seconds
            circuit_breaker_threshold: Consecutive retryable failures before cooldown
            circuit_breaker_cooldown: Cooldown seconds after the breaker opens
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or self._get_default_base_url(provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self.retry_max_backoff = max(self.retry_backoff, float(retry_max_backoff))
        self.circuit_breaker_threshold = max(1, int(circuit_breaker_threshold))
        self.circuit_breaker_cooldown = max(0.0, float(circuit_breaker_cooldown))
        self._consecutive_retryable_failures = 0
        self._circuit_open_until = 0.0
        self._client = None

    @staticmethod
    def _get_default_base_url(provider: str) -> str:
        bases = {
            "groq": "https://api.groq.com/openai/v1",
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
        }
        return bases.get(provider.lower(), "")

    def _get_client(self):
        """Get or create HTTP client."""
        if self._client is not None:
            return self._client

        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for API generation: pip install httpx")

        headers = {"Content-Type": "application/json"}

        if self.api_key:
            if self.provider == "anthropic":
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )
        return self._client

    def _is_retryable_error(self, error: Exception) -> bool:
        """Return True for rate limits and temporary API/network failures."""
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True

        try:
            import httpx
        except ImportError:
            return False

        return isinstance(error, (httpx.TimeoutException, httpx.TransportError))

    def _retry_delay(self, error: Exception, attempt: int) -> float:
        """Compute retry delay, honoring Retry-After when providers send it."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), self.retry_max_backoff)
            except ValueError:
                pass

        delay = self.retry_backoff * (2 ** max(0, attempt - 1))
        return min(delay, self.retry_max_backoff)

    def _raise_if_circuit_open(self) -> None:
        now = time.monotonic()
        if now < self._circuit_open_until:
            remaining = self._circuit_open_until - now
            raise APICircuitBreakerOpen(
                "API provider is temporarily rate limited or unavailable; "
                f"retry after {remaining:.1f}s."
            )

    def _record_success(self) -> None:
        self._consecutive_retryable_failures = 0
        self._circuit_open_until = 0.0

    def _record_retryable_failure(self) -> None:
        self._consecutive_retryable_failures += 1
        if self._consecutive_retryable_failures >= self.circuit_breaker_threshold:
            self._circuit_open_until = time.monotonic() + self.circuit_breaker_cooldown
            logger.warning(
                "API circuit breaker opened for %.1fs after %s consecutive failures",
                self.circuit_breaker_cooldown,
                self._consecutive_retryable_failures,
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response using the API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            Generated text
        """
        if not self.api_key:
            raise ValueError(
                f"API key required for {self.provider}. "
                f"Set {self.provider.upper()}_API_KEY environment variable."
            )

        self._raise_if_circuit_open()
        client = self._get_client()
        max_t = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        if self.provider == "anthropic":
            body = {
                "model": self.model,
                "messages": self._build_messages(prompt, system_prompt),
                "max_tokens": max_t,
                "temperature": temp,
            }
        else:
            body = {
                "model": self.model,
                "messages": self._build_messages(prompt, system_prompt),
                "max_tokens": max_t,
                "temperature": temp,
            }

        start = time.perf_counter()
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.post("/chat/completions", json=body)
                response.raise_for_status()
                data = response.json()
                elapsed = time.perf_counter() - start

                if self.provider == "anthropic":
                    text = data["content"][0]["text"]
                else:
                    text = data["choices"][0]["message"]["content"]

                logger.info(
                    f"API generation completed in {elapsed:.2f}s "
                    f"({data.get('usage', {}).get('total_tokens', '?')} tokens)"
                )
                self._record_success()
                return text.strip()

            except Exception as e:
                last_error = e
                retryable = self._is_retryable_error(e)
                if attempt >= self.max_retries or not retryable:
                    if retryable:
                        self._record_retryable_failure()
                    logger.error(f"API generation failed: {e}")
                    raise

                delay = self._retry_delay(e, attempt + 1)
                logger.warning(
                    "API generation failed with retryable error "
                    f"(attempt {attempt + 1}/{self.max_retries + 1}); "
                    f"retrying in {delay:.2f}s: {e}"
                )
                if delay > 0:
                    time.sleep(delay)

        raise last_error

    def generate_from_context(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ) -> str:
        """Generate response from question + retrieved context."""
        if user_template is None:
            user_template = (
                "Ngữ cảnh (Context):\n{context}\n\n"
                "Câu hỏi: {question}\n\nTrả lời:"
            )

        prompt = user_template.format(question=question, context=context)
        return self.generate(prompt, system_prompt=system_prompt)

    def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        callback=None,
        **kwargs,
    ) -> str:
        """Streaming generation using httpx streaming."""
        if not self.api_key:
            raise ValueError(f"API key required for {self.provider}")

        import httpx

        headers = {"Content-Type": "application/json"}
        if self.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model,
            "messages": self._build_messages(prompt, system_prompt),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        full_text = []

        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=body,
                headers=headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        import json
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                chunk = delta["content"]
                                full_text.append(chunk)
                                if callback:
                                    callback(chunk)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

            return "".join(full_text)

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> list[dict]:
        """Build message list for chat API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def load(self):
        """No-op for API generators (connection is established on first request)."""
        pass
