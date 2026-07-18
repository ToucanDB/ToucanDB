"""
ToucanDB Core Types and Data Structures

This module defines the fundamental types used throughout ToucanDB,
including vector representations, metadata structures, and search results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Type aliases for clarity
VectorData = list[float] | np.ndarray[Any, Any]
MetadataDict = dict[str, Any]
VectorId = str | int

T = TypeVar("T")


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
    HAMMING = "hamming"


class IndexType(str, Enum):
    """Supported vector index algorithms."""

    HNSW = "hnsw"  # Hierarchical Navigable Small World
    IVF = "ivf"  # Inverted File Index
    FLAT = "flat"  # Brute force (exact search)
    LSH = "lsh"  # Locality Sensitive Hashing
    ANNOY = "annoy"  # Approximate Nearest Neighbors Oh Yeah


class CompressionType(str, Enum):
    """Supported compression algorithms."""

    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"


class QuantizationType(str, Enum):
    """Supported vector quantization methods."""

    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"
    BINARY = "binary"
    PQ = "product_quantization"  # Product Quantization


@dataclass
class Vector:
    """Represents a single vector with metadata."""

    id: VectorId
    data: np.ndarray[Any, Any]
    metadata: MetadataDict
    timestamp: datetime

    def __post_init__(self) -> None:
        """Ensure vector data is a numpy array."""
        if not isinstance(self.data, np.ndarray):
            object.__setattr__(self, "data", np.array(self.data, dtype=np.float32))

    @property
    def dimensions(self) -> int:
        """Get the number of dimensions in the vector."""
        return len(self.data)

    def normalize(self) -> Vector:
        """Return a normalized copy of the vector."""
        norm = np.linalg.norm(self.data)
        if norm > 0:
            normalized_data = self.data / norm
        else:
            normalized_data = self.data.copy()

        return Vector(
            id=self.id,
            data=normalized_data,
            metadata=self.metadata.copy(),
            timestamp=self.timestamp,
        )


class VectorSchema(BaseModel):
    """Schema definition for a vector collection."""

    name: str = Field(..., description="Collection name")
    dimensions: int = Field(..., gt=0, description="Vector dimensions")
    metric: DistanceMetric = Field(default=DistanceMetric.COSINE)
    index_type: IndexType = Field(default=IndexType.HNSW)
    compression: CompressionType = Field(default=CompressionType.LZ4)
    quantization: QuantizationType = Field(default=QuantizationType.NONE)

    # Index-specific parameters
    hnsw_ef_construction: int = Field(default=200, gt=0)
    hnsw_m: int = Field(default=16, gt=0)
    ivf_nlist: int = Field(default=1024, gt=0)

    # Storage parameters
    max_vectors: int | None = Field(default=None, gt=0)
    enable_metadata_index: bool = Field(default=True)
    metadata_schema: dict[str, str] | None = Field(default=None)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Keep collection names portable and safe as directory names."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
            raise ValueError(
                "Collection names must be 1-128 characters and contain only "
                "letters, numbers, '.', '_' or '-'"
            )
        if value in {".", ".."}:
            raise ValueError("Collection name cannot be '.' or '..'")
        return value

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        if v > 10000:
            raise ValueError("Dimensions cannot exceed 10,000")
        return v

    @model_validator(mode="after")
    def validate_supported_configuration(self) -> VectorSchema:
        """Reject advertised-but-unimplemented combinations at creation time."""
        supported_metrics = {
            DistanceMetric.COSINE,
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.DOT_PRODUCT,
        }
        supported_indices = {IndexType.FLAT, IndexType.HNSW, IndexType.IVF}
        supported_compression = {CompressionType.NONE, CompressionType.LZ4}
        supported_quantization = {
            QuantizationType.NONE,
            QuantizationType.FP16,
            QuantizationType.INT8,
        }

        if self.metric not in supported_metrics:
            raise ValueError(f"Unsupported distance metric: {self.metric.value}")
        if self.index_type not in supported_indices:
            raise ValueError(f"Unsupported index type: {self.index_type.value}")
        if self.compression not in supported_compression:
            raise ValueError(f"Unsupported compression type: {self.compression.value}")
        if self.quantization not in supported_quantization:
            raise ValueError(
                f"Unsupported quantization type: {self.quantization.value}"
            )
        return self


@dataclass
class SearchResult:
    """Result from a vector similarity search."""

    id: VectorId
    vector: np.ndarray | None
    score: float
    metadata: MetadataDict
    distance: float

    def __lt__(self, other: SearchResult) -> bool:
        """Enable sorting by score (higher is better)."""
        return self.score > other.score


class SearchQuery(BaseModel):
    """Query specification for vector search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector: list[float] | np.ndarray[Any, Any] = Field(..., description="Query vector")
    k: int = Field(default=10, gt=0, le=1000, description="Number of results")
    threshold: float | None = Field(
        default=None,
        description=(
            "Minimum similarity score. Dot-product scores are not limited to "
            "the [0, 1] interval."
        ),
    )
    include_vectors: bool = Field(default=False)
    include_metadata: bool = Field(default=True)
    metadata_filter: dict[str, Any] | None = Field(default=None)

    # Search parameters
    ef: int | None = Field(default=None, gt=0)  # HNSW search parameter
    nprobe: int | None = Field(default=None, gt=0)  # IVF search parameter


class InsertRequest(BaseModel):
    """Request to insert vectors into a collection."""

    vectors: list[dict[str, Any]] = Field(..., min_length=1)
    batch_size: int = Field(default=1000, gt=0)
    upsert: bool = Field(default=False)

    @field_validator("vectors")
    @classmethod
    def validate_vectors(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, vec in enumerate(v):
            if "id" not in vec:
                raise ValueError(f"Vector {i} missing required field 'id'")
            if "vector" not in vec:
                raise ValueError(f"Vector {i} missing required field 'vector'")
        return v


@dataclass
class CollectionStats:
    """Statistics about a vector collection."""

    name: str
    total_vectors: int
    dimensions: int
    size_bytes: int
    index_type: IndexType
    metric: DistanceMetric
    created_at: datetime
    updated_at: datetime

    # Performance metrics
    avg_search_latency_ms: float
    cache_hit_ratio: float
    compression_ratio: float


class DatabaseConfig(BaseModel):
    """Configuration for ToucanDB instance."""

    class StorageConfig(BaseModel):
        path: str = Field(..., description="Database storage path")
        encryption_key: str | None = Field(default=None, exclude=True, repr=False)

    class MemoryConfig(BaseModel):
        cache_size_mb: int = Field(default=64, ge=0)

    class SecurityConfig(BaseModel):
        enable_encryption: bool = Field(default=False)

    class PerformanceConfig(BaseModel):
        index_snapshot_interval: int = Field(default=10_000, gt=0)
        compaction_min_tombstones: int = Field(default=64, gt=0)
        compaction_ratio: float = Field(default=0.20, gt=0.0, le=1.0)

    storage: StorageConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


class ErrorCode(str, Enum):
    """ToucanDB error codes."""

    COLLECTION_NOT_FOUND = "COLLECTION_NOT_FOUND"
    VECTOR_NOT_FOUND = "VECTOR_NOT_FOUND"
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    ENCRYPTION_ERROR = "ENCRYPTION_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"
    INDEX_ERROR = "INDEX_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


@dataclass
class OperationResult(Generic[T]):
    """Generic result wrapper for operations."""

    success: bool
    data: T | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    execution_time_ms: float = 0.0

    @classmethod
    def success_result(
        cls, data: T, execution_time_ms: float = 0.0
    ) -> OperationResult[T]:
        """Create a successful operation result."""
        return cls(success=True, data=data, execution_time_ms=execution_time_ms)

    @classmethod
    def error_result(
        cls, error_code: ErrorCode, error_message: str, execution_time_ms: float = 0.0
    ) -> OperationResult[T]:
        """Create an error operation result."""
        return cls(
            success=False,
            error_code=error_code,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
        )


# Batch operation types
class BatchOperation(BaseModel):
    """Base class for batch operations."""

    operation_type: Literal["insert", "update", "delete"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchInsert(BatchOperation):
    """Batch insert operation."""

    operation_type: Literal["insert"] = "insert"
    vectors: list[dict[str, Any]]


class BatchUpdate(BatchOperation):
    """Batch update operation."""

    operation_type: Literal["update"] = "update"
    updates: list[dict[str, Any]]


class BatchDelete(BatchOperation):
    """Batch delete operation."""

    operation_type: Literal["delete"] = "delete"
    ids: list[VectorId]


BatchOperationType = BatchInsert | BatchUpdate | BatchDelete
