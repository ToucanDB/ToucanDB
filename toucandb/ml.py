"""
ToucanDB ML Integration Module

This module provides machine learning integration capabilities for ToucanDB,
including embedding providers, text processing, and semantic search functionality.
"""

import asyncio
import hashlib
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import (
    Any,
    Protocol,
    cast,
    runtime_checkable,
)

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for embedding providers.

    All embedding providers must implement this interface to be compatible
    with ToucanDB's ML-first features.
    """

    @property
    def dimensions(self) -> int:
        """Number of values returned for each embedded text."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors (one per input text)
        """
        ...


class SimpleEmbeddingProvider:
    """
    A simple embedding provider for testing and development.

    This provider generates deterministic hash-seeded embeddings.
    For production use, replace with a real embedding model like sentence-transformers.
    """

    def __init__(self, dimensions: int = 384, seed: int | None = None):
        """
        Initialize the simple embedding provider.

        Args:
            dimensions: Dimensionality of the embedding vectors
            seed: Random seed for reproducible embeddings
        """
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.seed = seed

        logger.info(
            "SimpleEmbeddingProvider initialized with %s dimensions",
            dimensions,
        )

    @property
    def model_id(self) -> str:
        """Stable identifier used to invalidate stale document embeddings."""
        return f"toucandb-simple-v1:{self.dimensions}:{self.seed}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate simple hash-based embeddings for texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for text in texts:
            # Create a simple hash-based embedding
            # This is just for testing - use real embeddings in production
            # Python's built-in hash is randomized between processes. A stable
            # digest keeps persisted test/development indexes searchable after
            # the process restarts.
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
            text_seed = int.from_bytes(digest, "big")
            if self.seed is not None:
                text_seed ^= self.seed
            rng = np.random.default_rng(text_seed)

            # Generate normalized random vector
            embedding = rng.normal(0, 1, self.dimensions)
            embedding = embedding / np.linalg.norm(embedding)

            embeddings.append(embedding.tolist())

        logger.debug(f"Generated embeddings for {len(texts)} texts")
        return embeddings


class SentenceTransformerEmbeddingProvider:
    """Local sentence-transformers provider with lazy model loading.

    The model runs on the caller's machine, so source text does not need to be
    sent to a hosted embedding API. ``sentence-transformers`` is available via
    the ``toucandb[embeddings]`` optional dependency.
    """

    KNOWN_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        dimensions: int | None = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._dimensions = dimensions or self.KNOWN_DIMENSIONS.get(model_name)
        self._model: Any | None = None
        self._model_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return (
            f"sentence-transformers:{self.model_name}:"
            f"normalize={self.normalize_embeddings}"
        )

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            model_dimensions = self._get_model().get_sentence_embedding_dimension()
            if model_dimensions is None:
                raise ValueError(
                    f"Could not determine embedding dimensions for {self.model_name!r}"
                )
            self._dimensions = int(model_dimensions)
        return self._dimensions

    def _get_model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        from sentence_transformers import (  # type: ignore[import-not-found]
                            SentenceTransformer,
                        )
                    except ImportError as exc:
                        raise ImportError(
                            "SentenceTransformerEmbeddingProvider requires the "
                            "embeddings extra: pip install "
                            "'toucandb[embeddings]'"
                        ) from exc
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                    )
        return self._model

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._get_model().encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return cast(list[list[float]], encoded.astype(np.float32).tolist())

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed text without blocking the async event loop."""
        return await asyncio.to_thread(self._embed_sync, texts)


class CallableEmbeddingProvider:
    """Adapt a sync or async embedding callable without another dependency."""

    def __init__(
        self,
        embedder: Callable[
            [list[str]],
            list[list[float]] | Awaitable[list[list[float]]],
        ],
        *,
        dimensions: int,
        model_id: str,
    ):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")
        self._embedder = embedder
        self.dimensions = dimensions
        self.model_id = model_id

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._embedder(texts)
        if inspect.isawaitable(result):
            return await result
        return result


class OpenAIEmbeddingProvider:
    """Lazy async adapter for OpenAI's embeddings API.

    Install only when needed with ``pip install 'toucandb[openai]'``. A client
    may be injected for custom transports, testing, or centrally managed auth.
    """

    KNOWN_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        dimensions: int | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        batch_size: int = 256,
    ):
        inferred_dimensions = dimensions or self.KNOWN_DIMENSIONS.get(model)
        if inferred_dimensions is None or inferred_dimensions <= 0:
            raise ValueError(
                "dimensions is required for an embedding model whose size is unknown"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.dimensions = inferred_dimensions
        self.api_key = api_key
        self.batch_size = batch_size
        self._client = client

    @property
    def model_id(self) -> str:
        return f"openai:{self.model}:{self.dimensions}"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "OpenAIEmbeddingProvider requires: "
                    "pip install 'toucandb[openai]'"
                ) from exc
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        client = self._get_client()
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            arguments: dict[str, Any] = {"model": self.model, "input": batch}
            if self.model.startswith("text-embedding-3"):
                arguments["dimensions"] = self.dimensions
            response = await client.embeddings.create(**arguments)
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend([list(map(float, item.embedding)) for item in ordered])
        return embeddings


def embedding_provider_id(provider: EmbeddingProvider) -> str:
    """Return a stable best-effort identity for cache invalidation."""
    explicit = getattr(provider, "model_id", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    provider_type = type(provider)
    return (
        f"{provider_type.__module__}.{provider_type.__qualname__}:"
        f"{provider.dimensions}"
    )


# Export public API
__all__ = [
    "EmbeddingProvider",
    "CallableEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "SimpleEmbeddingProvider",
    "embedding_provider_id",
]
