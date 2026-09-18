"""
ToucanDB - A Secure, Efficient ML-First Vector Database Engine

This module provides the main interface for ToucanDB, allowing users to
create, manage, and query vector collections with state-of-the-art performance
and security features.
"""

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import (
    CollectionNotFoundError,
    ConfigurationError,
    EncryptionError,
    InvalidSchemaError,
    StorageError,
    ToucanDBException,
)
from .ml import (
    CallableEmbeddingProvider,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    SimpleEmbeddingProvider,
    embedding_provider_id,
)
from .schema import SchemaManager
from .types import (
    CollectionStats,
    CompressionType,
    DatabaseConfig,
    DistanceMetric,
    ErrorCode,
    IndexType,
    InsertRequest,
    OperationResult,
    QuantizationType,
    SearchQuery,
    VectorId,
    VectorSchema,
)
from .vector_engine import EncryptionEngine, VectorCollection

logger = logging.getLogger(__name__)

__version__ = "2.0.0"
__author__ = "Pierre-Henry Soria"
__email__ = "pierre@ph7.me"
__license__ = "MIT"
__description__ = "A secure, efficient ML-first vector database engine"


class _DatabaseLock:
    """Cross-process exclusive lock for one embedded database directory."""

    def __init__(self, path: Path):
        self.path = path
        self._file = open(path, "a+b")
        try:
            if os.name == "nt":  # pragma: no cover - exercised in Windows CI
                import msvcrt

                msvcrt_module: Any = msvcrt

                if path.stat().st_size == 0:
                    self._file.write(b"0")
                    self._file.flush()
                self._file.seek(0)
                msvcrt_module.locking(self._file.fileno(), msvcrt_module.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._file.seek(0)
            self._file.truncate()
            acquired_at = datetime.now(timezone.utc).isoformat()
            self._file.write(f"pid={os.getpid()} acquired={acquired_at}\n".encode())
            self._file.flush()
        except OSError as exc:
            self._file.close()
            raise StorageError(
                "lock",
                str(path),
                "Database is already open in another process or instance",
            ) from exc

    def release(self) -> None:
        if self._file.closed:
            return
        try:
            if os.name == "nt":  # pragma: no cover - exercised in Windows CI
                import msvcrt

                msvcrt_module: Any = msvcrt

                self._file.seek(0)
                msvcrt_module.locking(self._file.fileno(), msvcrt_module.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()


# Export public API
__all__ = [
    "ToucanDB",
    "VectorSchema",
    "SearchQuery",
    "InsertRequest",
    "DatabaseConfig",
    "DistanceMetric",
    "IndexType",
    "CompressionType",
    "QuantizationType",
    "ToucanDBException",
    "CollectionNotFoundError",
    "InvalidSchemaError",
    "ConfigurationError",
    "ErrorCode",
    "EmbeddingProvider",
    "CallableEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "SimpleEmbeddingProvider",
]


class ToucanDB:
    """
    Main ToucanDB database interface.

    Provides a high-level API for creating and managing vector collections,
    with built-in security, compression, and performance optimizations.
    """

    def __init__(self, storage_path: str | Path, config: DatabaseConfig | None = None):
        """
        Initialize ToucanDB instance.

        Args:
            storage_path: Path to database storage directory
            config: Database configuration (optional)
        """
        self.storage_path = Path(storage_path).expanduser().resolve()
        self.config = (config or self._default_config()).model_copy(deep=True)
        self.config.storage.path = str(self.storage_path)
        if self.config.storage.encryption_key:
            self.config.security.enable_encryption = True
        if (
            self.config.security.enable_encryption
            and not self.config.storage.encryption_key
        ):
            raise ConfigurationError(
                "security.enable_encryption",
                True,
                "An encryption key is required when encryption is enabled",
            )

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._database_lock = _DatabaseLock(self.storage_path / ".toucandb.lock")

        try:
            # Initialize components
            self._encryption_engine = EncryptionEngine(
                self.config.storage.encryption_key,
                self.storage_path / "encryption.salt",
            )
            self.schema_manager = SchemaManager(self.storage_path)
            self.collections: dict[str, VectorCollection] = {}

            # Database metadata
            self.created_at = datetime.now(timezone.utc)
            self.last_backup: datetime | None = None

            # Load existing collections
            self._load_collections()

            self._embedding_provider: EmbeddingProvider | None = None
            self._closed = False
        except Exception:
            self._database_lock.release()
            raise

        logger.info("ToucanDB initialized at %s", self.storage_path)

    @classmethod
    async def create(
        cls,
        storage_path: str | Path,
        config: DatabaseConfig | None = None,
        encryption_key: str | None = None,
    ) -> "ToucanDB":
        """
        Create a new ToucanDB instance asynchronously.

        Args:
            storage_path: Path to database storage directory
            config: Database configuration
            encryption_key: Encryption key for data security

        Returns:
            ToucanDB instance
        """
        config = (config or cls._default_config()).model_copy(deep=True)
        if encryption_key is not None:
            config.storage.encryption_key = encryption_key
            config.security.enable_encryption = True

        db = cls(storage_path, config)
        await db._initialize_async()
        return db

    @staticmethod
    def _default_config() -> DatabaseConfig:
        """Create default database configuration."""
        return DatabaseConfig(
            storage=DatabaseConfig.StorageConfig(path="./toucandb_data")
        )

    async def _initialize_async(self) -> None:
        """Perform async initialization tasks."""
        # Future: Initialize background tasks, connections, etc.
        pass

    def _load_collections(self) -> None:
        """Load existing collections from storage."""
        collection_names = self.schema_manager.list_collections()
        cache_budget = self.config.memory.cache_size_mb * 1024 * 1024
        per_collection_cache = cache_budget // max(1, len(collection_names))

        for collection_name in collection_names:
            schema = self.schema_manager.get_schema(collection_name)
            if schema:
                try:
                    collection = VectorCollection(
                        name=collection_name,
                        schema=schema,
                        storage_path=self.storage_path,
                        encryption_key=self.config.storage.encryption_key,
                        cache_size_bytes=per_collection_cache,
                        encryption_engine=self._encryption_engine,
                        index_snapshot_interval=(
                            self.config.performance.index_snapshot_interval
                        ),
                        compaction_min_tombstones=(
                            self.config.performance.compaction_min_tombstones
                        ),
                        compaction_ratio=self.config.performance.compaction_ratio,
                    )
                    self.collections[collection_name] = collection
                    logger.info("Loaded collection: %s", collection_name)
                except EncryptionError:
                    for loaded in self.collections.values():
                        loaded.close()
                    raise
                except Exception as e:
                    for loaded in self.collections.values():
                        loaded.close()
                    raise StorageError(
                        "load_collection",
                        collection_name,
                        str(e),
                    ) from e

    def _rebalance_cache_budgets(self) -> None:
        """Share the configured database cache ceiling across collections."""
        if not self.collections:
            return
        total = self.config.memory.cache_size_mb * 1024 * 1024
        per_collection = total // len(self.collections)
        for collection in self.collections.values():
            collection.storage.set_cache_limit(per_collection)

    async def create_collection(
        self, schema: VectorSchema, overwrite: bool = False
    ) -> VectorCollection:
        """
        Create a new vector collection.

        Args:
            schema: Collection schema definition
            overwrite: Whether to overwrite existing collection

        Returns:
            VectorCollection instance

        Raises:
            InvalidSchemaError: If schema is invalid
            CollectionExistsError: If collection exists and overwrite=False
        """
        try:
            if schema.name in self.collections:
                if not overwrite:
                    raise InvalidSchemaError(
                        f"Collection {schema.name!r} already exists"
                    )
                await self.drop_collection(schema.name)

            inactive_schema = self.schema_manager.get_schema_version(schema.name)
            if inactive_schema is not None and not inactive_schema.is_active:
                stale_path = self.storage_path / schema.name
                if stale_path.exists():
                    await asyncio.to_thread(shutil.rmtree, stale_path)
                self.schema_manager.delete_schema(schema.name, purge=True)

            # Create schema
            self.schema_manager.create_schema(schema.name, schema, overwrite=False)

            # Create collection
            collection = VectorCollection(
                name=schema.name,
                schema=schema,
                storage_path=self.storage_path,
                encryption_key=self.config.storage.encryption_key,
                cache_size_bytes=self.config.memory.cache_size_mb * 1024 * 1024,
                encryption_engine=self._encryption_engine,
                index_snapshot_interval=(
                    self.config.performance.index_snapshot_interval
                ),
                compaction_min_tombstones=(
                    self.config.performance.compaction_min_tombstones
                ),
                compaction_ratio=self.config.performance.compaction_ratio,
            )

            self.collections[schema.name] = collection

            self._rebalance_cache_budgets()
            logger.info("Created collection: %s", schema.name)
            return collection

        except Exception as e:
            logger.error("Failed to create collection %s: %s", schema.name, e)
            raise

    def get_collection(self, name: str) -> VectorCollection:
        """
        Get an existing collection.

        Args:
            name: Collection name

        Returns:
            VectorCollection instance

        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        if name not in self.collections:
            raise CollectionNotFoundError(name)

        return self.collections[name]

    def list_collections(self) -> list[str]:
        """
        List all collections in the database.

        Returns:
            List of collection names
        """
        return list(self.collections.keys())

    def list_vector_ids(self, collection_name: str) -> list[VectorId]:
        """List the logical vector IDs currently present in a collection."""
        collection = self.get_collection(collection_name)
        return list(collection.index.reverse_mapping.keys())

    async def drop_collection(self, name: str) -> bool:
        """
        Drop a collection and all its data.

        Args:
            name: Collection name

        Returns:
            True if collection was dropped, False if it didn't exist
        """
        if name not in self.collections:
            return False

        try:
            collection = self.collections[name]
            await asyncio.to_thread(collection.close)
            await asyncio.to_thread(shutil.rmtree, collection.storage_path)
            self.schema_manager.delete_schema(name, purge=True)
            del self.collections[name]
            self._rebalance_cache_budgets()
            logger.info("Dropped collection: %s", name)
            return True

        except Exception as e:
            logger.error(f"Failed to drop collection {name}: {e}")
            raise StorageError("drop_collection", name, str(e)) from e

    async def insert_vectors(
        self,
        collection_name: str,
        vectors: list[dict[str, Any]],
        batch_size: int = 1000,
        *,
        upsert: bool = False,
    ) -> OperationResult[list[VectorId]]:
        """
        Insert vectors into a collection.

        Args:
            collection_name: Target collection name
            vectors: List of vector data dictionaries
            batch_size: Batch size for processing

        Returns:
            OperationResult with inserted vector IDs
        """
        collection = self.get_collection(collection_name)

        if batch_size <= 0:
            return OperationResult.error_result(
                ErrorCode.VALIDATION_ERROR,
                "batch_size must be positive",
            )

        # Validate the entire logical operation before the first batch commits,
        # preventing partial inserts for malformed input or cross-batch IDs.
        # This converts and checks every value with NumPy in a worker thread;
        # the prepared vectors are then inserted without being re-validated.
        prepared = await asyncio.to_thread(collection.prepare_vectors, vectors)
        if not prepared.success or prepared.data is None:
            return OperationResult.error_result(
                prepared.error_code or ErrorCode.VALIDATION_ERROR,
                prepared.error_message or "Invalid vectors",
                prepared.execution_time_ms,
            )
        prepared_vectors = prepared.data
        seen_ids = {vector.id for vector in prepared_vectors}

        existing_ids = seen_ids.intersection(collection.index.reverse_mapping)
        if existing_ids and not upsert:
            duplicate = sorted(str(value) for value in existing_ids)[0]
            return OperationResult.error_result(
                ErrorCode.VALIDATION_ERROR,
                f"Vector ID already exists: {duplicate!r}; use upsert=True",
            )
        projected_count = (
            collection.index.active_count + len(seen_ids) - len(existing_ids)
        )
        if (
            collection.schema.max_vectors is not None
            and projected_count > collection.schema.max_vectors
        ):
            return OperationResult.error_result(
                ErrorCode.VALIDATION_ERROR,
                f"Collection limit of {collection.schema.max_vectors} "
                "vectors exceeded",
            )

        # Process in batches for large datasets
        all_ids: list[VectorId] = []
        total_time = prepared.execution_time_ms

        for i in range(0, len(prepared_vectors), batch_size):
            batch = prepared_vectors[i : i + batch_size]
            result = await collection._insert_prepared(batch, upsert=upsert)

            if not result.success:
                return OperationResult.error_result(
                    result.error_code or ErrorCode.STORAGE_ERROR,
                    result.error_message or "Insert failed",
                    total_time,
                )

            if result.data:
                all_ids.extend(result.data)
            total_time += result.execution_time_ms

        return OperationResult.success_result(
            all_ids, total_time
        )  # type: ignore[arg-type]

    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: list[dict[str, Any]],
        batch_size: int = 1000,
    ) -> OperationResult[list[VectorId]]:
        """Insert new vectors and replace vectors with matching IDs."""
        return await self.insert_vectors(
            collection_name,
            vectors,
            batch_size=batch_size,
            upsert=True,
        )

    async def search_vectors(
        self, collection_name: str, query: SearchQuery
    ) -> OperationResult[list[dict[str, Any]]]:
        """
        Search for similar vectors in a collection.

        Args:
            collection_name: Target collection name
            query: Search query parameters

        Returns:
            OperationResult with search results
        """
        collection = self.get_collection(collection_name)
        result = await collection.search(query)

        if result.success:
            # Convert SearchResult objects to dictionaries (may be empty list)
            search_results: list[dict[str, Any]] = []
            for sr in result.data or []:
                result_dict: dict[str, Any] = {
                    "id": sr.id,
                    "score": sr.score,
                    "distance": sr.distance,
                }

                if query.include_metadata:
                    result_dict["metadata"] = sr.metadata

                if query.include_vectors:
                    result_dict["vector"] = (
                        sr.vector.tolist() if sr.vector is not None else None
                    )

                search_results.append(result_dict)

            return OperationResult.success_result(
                search_results, result.execution_time_ms
            )

        # Propagate error result with proper type
        return OperationResult[list[dict[str, Any]]](
            success=False,
            data=[],
            execution_time_ms=result.execution_time_ms,
            error_code=result.error_code,
            error_message=result.error_message,
        )

    async def delete_vector(
        self, collection_name: str, vector_id: VectorId
    ) -> OperationResult[bool]:
        """
        Delete a vector from a collection.

        Args:
            collection_name: Target collection name
            vector_id: ID of vector to delete

        Returns:
            OperationResult indicating success
        """
        collection = self.get_collection(collection_name)
        return await collection.delete(vector_id)

    async def delete_vectors(
        self,
        collection_name: str,
        vector_ids: Sequence[VectorId],
    ) -> OperationResult[int]:
        """Delete many vectors with a single storage transaction."""
        collection = self.get_collection(collection_name)
        return await collection.delete_many(vector_ids)

    def get_collection_stats(self, collection_name: str) -> CollectionStats:
        """
        Get statistics for a collection.

        Args:
            collection_name: Target collection name

        Returns:
            CollectionStats object
        """
        collection = self.get_collection(collection_name)
        return collection.get_stats()

    def get_database_info(self) -> dict[str, Any]:
        """
        Get comprehensive database information.

        Returns:
            Dictionary with database metadata and statistics
        """
        total_vectors = 0
        total_size = 0

        collection_info = {}
        for name, collection in self.collections.items():
            stats = collection.get_stats()
            collection_info[name] = {
                "vectors": stats.total_vectors,
                "size_bytes": stats.size_bytes,
                "dimensions": stats.dimensions,
                "index_type": stats.index_type,
                "metric": stats.metric,
            }
            total_vectors += stats.total_vectors
            total_size += stats.size_bytes

        return {
            "version": __version__,
            "storage_path": str(self.storage_path),
            "created_at": self.created_at.isoformat(),
            "total_collections": len(self.collections),
            "total_vectors": total_vectors,
            "total_size_bytes": total_size,
            "config": self.config.model_dump(),
            "collections": collection_info,
        }

    async def backup(self, backup_path: str | Path) -> bool:
        """
        Create a backup of the database.

        Args:
            backup_path: Path for backup files

        Returns:
            True if backup was successful
        """
        destination = Path(backup_path).expanduser().resolve()
        if destination == self.storage_path or self.storage_path in destination.parents:
            raise ValueError(
                "Backup destination must be outside the database directory"
            )
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError("Backup destination must be absent or empty")

        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            await asyncio.to_thread(temporary.mkdir, parents=True, exist_ok=False)
            await asyncio.to_thread(
                shutil.copytree,
                self.schema_manager.schemas_path,
                temporary / "schemas",
            )
            encryption_salt = self.storage_path / "encryption.salt"
            if encryption_salt.exists():
                await asyncio.to_thread(
                    shutil.copy2,
                    encryption_salt,
                    temporary / encryption_salt.name,
                )
            for name, collection in self.collections.items():
                await asyncio.to_thread(collection.backup_to, temporary / name)
            manifest = {
                "format": 1,
                "toucandb_version": __version__,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "collections": sorted(self.collections),
            }
            (temporary / "backup.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            if destination.exists():
                destination.rmdir()
            os.replace(temporary, destination)
            self.last_backup = datetime.now(timezone.utc)
            logger.info("Database backed up to %s", destination)
            return True
        except Exception:
            if temporary.exists():
                await asyncio.to_thread(shutil.rmtree, temporary)
            raise

    async def close(self) -> None:
        """
        Close the database and clean up resources.
        """
        if self._closed:
            return
        self._closed = True
        logger.info("Closing ToucanDB")
        try:
            await asyncio.gather(
                *(
                    asyncio.to_thread(collection.close)
                    for collection in self.collections.values()
                )
            )
        finally:
            self._database_lock.release()
        logger.info("ToucanDB closed successfully")

    # ML-first vector database methods
    def set_embedding_provider(self, provider: EmbeddingProvider) -> None:
        """Set the embedding provider for ML-first operations."""
        if not hasattr(provider, "embed") or not callable(provider.embed):
            raise ValueError("Provider must implement the EmbeddingProvider protocol")
        dimensions = getattr(provider, "dimensions", None)
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError(
                "Embedding provider must expose a positive integer "
                "dimensions property"
            )
        self._embedding_provider = provider
        logger.info("Embedding provider set successfully")

    @property
    def is_ml_ready(self) -> bool:
        """Check if the database is ready for ML operations."""
        return self._embedding_provider is not None

    @property
    def embedding_provider(self) -> EmbeddingProvider | None:
        """The configured provider, or ``None`` when document APIs are disabled."""
        return self._embedding_provider

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if self._embedding_provider is None:
            raise ValueError(
                "No embedding provider configured. Use set_embedding_provider() first."
            )
        embeddings: list[list[float]] = await self._embedding_provider.embed(texts)
        if len(embeddings) != len(texts):
            raise ValueError(
                "Embedding provider returned a different number of vectors than texts"
            )
        dimensions = self._embedding_provider.dimensions
        for index, embedding in enumerate(embeddings):
            if len(embedding) != dimensions:
                raise ValueError(
                    f"Embedding {index} has {len(embedding)} values; "
                    f"expected {dimensions}"
                )
            if not np.isfinite(np.asarray(embedding, dtype=np.float64)).all():
                raise ValueError(f"Embedding {index} contains a non-finite value")
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query text."""
        embeddings = await self.embed_texts([query])
        return embeddings[0]

    async def ensure_document_collection(
        self,
        name: str,
        *,
        dimensions: int | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        index_type: IndexType = IndexType.HNSW,
        metadata_schema: dict[str, str] | None = None,
    ) -> VectorCollection:
        """Return a document collection, creating it when necessary.

        When ``dimensions`` is omitted, the configured embedding provider's
        declared dimensions are used. Existing collections are checked to
        prevent silently querying them with a different embedding model.
        """
        if dimensions is None:
            if not self.is_ml_ready:
                raise ValueError(
                    "No dimensions supplied and no embedding provider configured"
                )
            provider_dimensions = getattr(self._embedding_provider, "dimensions", None)
            if not isinstance(provider_dimensions, int) or provider_dimensions <= 0:
                raise ValueError(
                    "Embedding provider must expose a positive integer "
                    "dimensions property"
                )
            dimensions = provider_dimensions

        if name in self.collections:
            collection = self.collections[name]
            if collection.schema.dimensions != dimensions:
                raise InvalidSchemaError(
                    f"Collection {name!r} has {collection.schema.dimensions} "
                    f"dimensions, but the provider uses {dimensions}"
                )
            return collection

        return await self.create_collection(
            VectorSchema(
                name=name,
                dimensions=dimensions,
                metric=metric,
                index_type=index_type,
                metadata_schema=metadata_schema,
            )
        )

    async def insert_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        *,
        upsert: bool = False,
    ) -> OperationResult[list[VectorId]]:
        """Insert documents, avoiding repeat embeddings during idempotent upserts."""
        started = time.perf_counter()
        if ids is not None and len(ids) != len(documents):
            raise ValueError("ids must contain exactly one value per document")
        if metadata is not None and len(metadata) != len(documents):
            raise ValueError("metadata must contain exactly one value per document")

        if self._embedding_provider is None:
            raise ValueError(
                "No embedding provider configured. Use set_embedding_provider() first."
            )
        collection = self.get_collection(collection_name)
        if collection.schema.dimensions != self._embedding_provider.dimensions:
            raise InvalidSchemaError(
                f"Collection {collection_name!r} has {collection.schema.dimensions} "
                f"dimensions, but the provider uses "
                f"{self._embedding_provider.dimensions}"
            )

        vector_ids: list[VectorId] = (
            list(ids)
            if ids is not None
            else [f"doc_{index}" for index in range(len(documents))]
        )
        if len(set(vector_ids)) != len(vector_ids):
            return OperationResult.error_result(
                ErrorCode.VALIDATION_ERROR,
                "Document IDs must be unique within a batch",
                (time.perf_counter() - started) * 1000,
            )
        provider_id = embedding_provider_id(self._embedding_provider)
        desired_metadata: list[dict[str, Any]] = []
        changed_indices: list[int] = []
        # One batched read off the event loop instead of a query per document.
        persisted_vectors = await asyncio.to_thread(
            collection.storage.load_vectors, vector_ids
        )
        for index, document in enumerate(documents):
            vector_metadata = dict(metadata[index]) if metadata else {}
            vector_metadata["document"] = document
            vector_metadata["_toucandb_embedding_provider"] = provider_id
            desired_metadata.append(vector_metadata)
            persisted = persisted_vectors.get(vector_ids[index])
            if persisted is not None:
                if not upsert:
                    return OperationResult.error_result(
                        ErrorCode.VALIDATION_ERROR,
                        f"Vector ID already exists: {vector_ids[index]!r}; "
                        "use upsert=True",
                        (time.perf_counter() - started) * 1000,
                    )
                if persisted.metadata == vector_metadata:
                    continue
            changed_indices.append(index)

        if not changed_indices:
            return OperationResult.success_result(
                vector_ids,
                (time.perf_counter() - started) * 1000,
            )

        changed_documents = [documents[index] for index in changed_indices]
        embeddings = await self.embed_texts(changed_documents)
        vectors: list[dict[str, Any]] = []
        for changed_offset, index in enumerate(changed_indices):
            vectors.append(
                {
                    "id": vector_ids[index],
                    "vector": embeddings[changed_offset],
                    "metadata": desired_metadata[index],
                }
            )
        result = await self.insert_vectors(collection_name, vectors, upsert=upsert)
        if not result.success:
            return result
        return OperationResult.success_result(
            vector_ids,
            (time.perf_counter() - started) * 1000,
        )

    async def upsert_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> OperationResult[list[VectorId]]:
        """Embed documents and replace any existing documents with matching IDs."""
        return await self.insert_documents(
            collection_name,
            documents,
            metadata=metadata,
            ids=ids,
            upsert=True,
        )

    async def semantic_search(
        self,
        collection_name: str,
        query: str,
        k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform semantic search using query embedding."""
        query_embedding = await self.embed_query(query)
        search_query = SearchQuery(
            vector=query_embedding,
            k=k,
            metadata_filter=filter_metadata,
            include_metadata=True,
        )
        result = await self.search_vectors(collection_name, search_query)

        if not result.success:
            raise RuntimeError(result.error_message or "Semantic search failed")
        if not result.data:
            return []

        formatted_results: list[dict[str, Any]] = []
        for item in result.data:
            item_dict = item if isinstance(item, dict) else item.__dict__
            metadata = item_dict.get("metadata", {})
            formatted_result: dict[str, Any] = {
                "id": item_dict.get("id"),
                "score": item_dict.get("score"),
                "document": (
                    metadata.get("document", "") if isinstance(metadata, dict) else ""
                ),
                "metadata": (
                    {
                        key: value
                        for key, value in metadata.items()
                        if key != "document" and not key.startswith("_toucandb_")
                    }
                    if isinstance(metadata, dict)
                    else {}
                ),
            }
            formatted_results.append(formatted_result)
        return formatted_results

    async def __aenter__(self) -> "ToucanDB":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


# Convenience functions
async def create_database(
    storage_path: str | Path, encryption_key: str | None = None
) -> ToucanDB:
    """
    Create a new ToucanDB database.

    Args:
        storage_path: Path to database storage
        encryption_key: Optional encryption key

    Returns:
        ToucanDB instance
    """
    return await ToucanDB.create(storage_path, encryption_key=encryption_key)


def create_schema(
    name: str,
    dimensions: int,
    metric: str = "cosine",
    index_type: str = "hnsw",
    **kwargs: Any,
) -> VectorSchema:
    """
    Create a vector schema with sensible defaults.

    Args:
        name: Collection name
        dimensions: Vector dimensions
        metric: Distance metric
        index_type: Index algorithm
        **kwargs: Additional schema parameters

    Returns:
        VectorSchema instance
    """
    return VectorSchema(
        name=name,
        dimensions=dimensions,
        metric=DistanceMetric(metric),
        index_type=IndexType(index_type),
        **kwargs,
    )
