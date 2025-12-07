"""
ToucanDB Core Types and Data Structures

This module defines the fundamental types used throughout ToucanDB,
including vector representations, metadata structures, and search results.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Literal, Optional, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Field, validator

# Type aliases for clarity
VectorData = Union[list[float], np.ndarray]
MetadataDict = dict[str, Any]
VectorId = Union[str, int]

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
    data: np.ndarray
    metadata: MetadataDict
    timestamp: datetime

    def __post_init__(self):
        """Ensure vector data is a numpy array."""
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.float32)

    @property
    def dimensions(self) -> int:
        """Get the number of dimensions in the vector."""
        return len(self.data)

    def normalize(self) -> "Vector":
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
    max_vectors: Optional[int] = Field(default=None, gt=0)
    enable_metadata_index: bool = Field(default=True)
    metadata_schema: Optional[dict[str, str]] = Field(default=None)

    @validator("dimensions")
    def validate_dimensions(cls, v):
        if v > 10000:
            raise ValueError("Dimensions cannot exceed 10,000")
        return v


@dataclass
class SearchResult:
    """Result from a vector similarity search."""

    id: VectorId
    vector: Optional[np.ndarray]
    score: float
    metadata: MetadataDict
    distance: float

    def __lt__(self, other: "SearchResult") -> bool:
        """Enable sorting by score (higher is better)."""
        return self.score > other.score


class SearchQuery(BaseModel):
    """Query specification for vector search."""

    vector: list[float] = Field(..., description="Query vector")
    k: int = Field(default=10, gt=0, le=1000, description="Number of results")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    include_vectors: bool = Field(default=False)
    include_metadata: bool = Field(default=True)
    metadata_filter: Optional[dict[str, Any]] = Field(default=None)

    # Search parameters
    ef: Optional[int] = Field(default=None, gt=0)  # HNSW search parameter
    nprobe: Optional[int] = Field(default=None, gt=0)  # IVF search parameter


class InsertRequest(BaseModel):
    """Request to insert vectors into a collection."""

    vectors: list[dict[str, Any]] = Field(..., min_items=1)
    batch_size: int = Field(default=1000, gt=0)
    upsert: bool = Field(default=False)

    @validator("vectors")
    def validate_vectors(cls, v):
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
        compression: CompressionType = Field(default=CompressionType.LZ4)
        encryption_key: Optional[str] = Field(default=None)
        backup_interval_seconds: int = Field(default=3600, gt=0)
        max_file_size_mb: int = Field(default=1024, gt=0)

    class MemoryConfig(BaseModel):
        cache_size_mb: int = Field(default=512, gt=0)
        enable_memory_mapping: bool = Field(default=True)
        preload_collections: bool = Field(default=False)
        gc_threshold: float = Field(default=0.8, gt=0.0, le=1.0)

    class SecurityConfig(BaseModel):
        enable_encryption: bool = Field(default=True)
        enable_audit_logging: bool = Field(default=True)
        api_key_required: bool = Field(default=False)
        max_connections: int = Field(default=100, gt=0)

    class PerformanceConfig(BaseModel):
        num_workers: int = Field(default=4, gt=0)
        enable_simd: bool = Field(default=True)
        batch_size: int = Field(default=1000, gt=0)
        auto_optimize_indices: bool = Field(default=True)

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
    data: Optional[T] = None
    error_code: Optional[ErrorCode] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0

    @classmethod
    def success_result(
        cls, data: T, execution_time_ms: float = 0.0
    ) -> "OperationResult[T]":
        """Create a successful operation result."""
        return cls(success=True, data=data, execution_time_ms=execution_time_ms)

    @classmethod
    def error_result(
        cls, error_code: ErrorCode, error_message: str, execution_time_ms: float = 0.0
    ) -> "OperationResult[T]":
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
    timestamp: datetime = Field(default_factory=datetime.utcnow)


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


BatchOperationType = Union[BatchInsert, BatchUpdate, BatchDelete]
