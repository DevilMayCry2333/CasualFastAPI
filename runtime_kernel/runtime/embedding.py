"""
embedding — EmbeddingClient: independent embedding API wrapper.

Completely separate from LLMClient. Calls a dedicated embedding endpoint
at a configurable address, compatible with OpenAI Embedding API.

Example:
    client = EmbeddingClient(
        api_url="http://0.0.0.0:28001/v1/embeddings",
        model="text-embedding-ada-002",
    )
    vector = client.embed("Some text to embed")
"""

from __future__ import annotations

from typing import Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

from runtime_kernel.runtime.exceptions import EmbeddingError, ConfigurationError


class EmbeddingClient:
    """Encapsulates all embedding API interactions.

    Calls a separate embedding endpoint (not the LLM endpoint).
    Follows the OpenAI /v1/embeddings API contract.
    """

    def __init__(
        self,
        api_url: str = "http://0.0.0.0:28001/v1/embeddings",
        model: str = "text-embedding-ada-002",
        api_key: str = "",
        timeout: int = 60,
    ) -> None:
        if requests is None:
            raise ConfigurationError(
                "The 'requests' library is required. Install with: pip install requests"
            )

        self.api_url = api_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed(self, text: str) -> list[float]:
        """Embed a single text string and return the embedding vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
            Returns an empty list if the API call fails.

        Raises:
            EmbeddingError: If the API returns an error status.
        """
        if not text or not text.strip():
            return []

        payload = {
            "model": self.model,
            "input": text,
        }

        try:
            r = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                headers=self._build_headers(),
                proxies={"http": None, "https": None},
            )
            if r.status_code != 200:
                raise EmbeddingError(
                    f"Embedding API returned {r.status_code}: {r.text[:200]}"
                )

            data = r.json()
            return data.get("data", [{}])[0].get("embedding", [])

        except requests.exceptions.Timeout:
            raise EmbeddingError(
                f"Embedding request timed out after {self.timeout}s"
            )
        except requests.exceptions.ConnectionError as e:
            raise EmbeddingError(f"Embedding connection failed: {e}")
        except Exception as e:
            raise EmbeddingError(f"Embedding request failed: {e}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            r = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout * 2,
                headers=self._build_headers(),
                proxies={"http": None, "https": None},
            )
            if r.status_code != 200:
                raise EmbeddingError(
                    f"Embedding API returned {r.status_code}: {r.text[:200]}"
                )

            data = r.json()
            results = data.get("data", [])
            # Sort by index to preserve order
            results.sort(key=lambda x: x.get("index", 0))
            return [item.get("embedding", []) for item in results]

        except requests.exceptions.Timeout:
            raise EmbeddingError(
                f"Embedding batch request timed out after {self.timeout * 2}s"
            )
        except requests.exceptions.ConnectionError as e:
            raise EmbeddingError(f"Embedding batch connection failed: {e}")
        except Exception as e:
            raise EmbeddingError(f"Embedding batch request failed: {e}")

    def health_check(self) -> bool:
        """Check if the embedding API is reachable.

        Returns True if healthy.
        """
        try:
            result = self.embed("ping")
            return len(result) > 0
        except EmbeddingError:
            return False
