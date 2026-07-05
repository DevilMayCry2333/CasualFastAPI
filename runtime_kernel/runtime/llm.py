"""
llm — LLMClient: unified interface for LLM calls.

Supports:
  - OpenAI
  - DeepSeek
  - OpenAI Compatible API

Methods:
  - complete() — synchronous text completion
  - stream() — streaming text completion (generator)
  - health_check() — verify API connectivity

The Runtime Engine never calls requests.post() directly.
"""

from __future__ import annotations

import json
from typing import Any, Generator, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

from runtime_kernel.runtime.exceptions import LLMError, ConfigurationError


DEFAULT_TIMEOUT = 300  # seconds


def _sanitize(text: str) -> str:
    """Remove surrogate characters that cannot be UTF-8 encoded."""
    if not text:
        return text
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Sanitize all message content to ensure valid UTF-8."""
    return [
        {
            "role": m.get("role", "user"),
            "content": _sanitize(m.get("content", "")),
        }
        for m in messages
    ]


class LLMClient:
    """Encapsulates all LLM API interactions."""

    def __init__(
        self,
        api_url: str = "http://localhost:28000/v1/chat/completions",
        model: str = "deepseek-v4-flash",
        api_key: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        temperature: float = 0.85,
        max_tokens: int = 4096,
    ) -> None:
        if requests is None:
            raise ConfigurationError(
                "The 'requests' library is required. Install with: pip install requests"
            )

        self.api_url = api_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> dict:
        return {
            "model": self.model,
            "messages": _sanitize_messages(messages),
            "temperature": round(temperature if temperature is not None else self.temperature, 2),
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": stream,
        }

    def complete(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Send a completion request and return the response text.

        Returns None on error.
        """
        payload = self._build_payload(messages, temperature, max_tokens, stream=False)
        try:
            r = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                headers=self._build_headers(),
                proxies={"http": None, "https": None},
            )
            if r.status_code != 200:
                raise LLMError(f"API returned {r.status_code}: {r.text[:200]}")

            data = r.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if content:
                content = _sanitize(content.strip())
            return content or None

        except requests.exceptions.Timeout:
            raise LLMError(f"LLM request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise LLMError(f"LLM connection failed: {e}")
        except Exception as e:
            raise LLMError(f"LLM request failed: {e}")

    def stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Stream a completion response, yielding text chunks.

        Yields:
            Text chunks as they arrive.
        """
        payload = self._build_payload(messages, temperature, max_tokens, stream=True)
        try:
            with requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                headers=self._build_headers(),
                stream=True,
                proxies={"http": None, "https": None},
            ) as r:
                if r.status_code != 200:
                    raise LLMError(f"API returned {r.status_code}: {r.text[:200]}")

                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        chunk = line[6:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                            delta = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                yield _sanitize(delta)
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.Timeout:
            raise LLMError(f"LLM stream timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise LLMError(f"LLM stream connection failed: {e}")
        except Exception as e:
            raise LLMError(f"LLM stream failed: {e}")

    def health_check(self) -> bool:
        """Check if the LLM API is reachable and responsive.

        Sends a minimal request and returns True if healthy.
        """
        try:
            result = self.complete(
                [{"role": "user", "content": "ping"}],
                temperature=0.1,
                max_tokens=5,
            )
            return result is not None
        except LLMError:
            return False

    def update_config(
        self,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Update configuration at runtime."""
        if api_url is not None:
            self.api_url = api_url.rstrip("/")
        if model is not None:
            self.model = model
        if api_key is not None:
            self.api_key = api_key
        if timeout is not None:
            self.timeout = timeout
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
