"""
ToucanDB Vector Engine

The core vector processing engine that handles indexing, search, and storage
of high-dimensional vectors with state-of-the-art algorithms and optimizations.
"""

import os
import time
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Union, Any, AsyncIterator
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging

try:
    import numpy as np
    import faiss
    from scipy.spatial.distance import cosine, euclidean
    import lz4.frame
    import xxhash
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import msgpack
except ImportError as e:
    # These imports will be available after installing dependencies
    pass

from .types import (
    Vector, VectorSchema, SearchResult, SearchQuery, InsertRequest,
    DistanceMetric, IndexType, CompressionType, QuantizationType,
    VectorId, MetadataDict, OperationResult, ErrorCode, CollectionStats
)
from .exceptions import (
    ToucanDBException, IndexError, MemoryError, StorageError,
    DimensionMismatchError, EncryptionError, ValidationError
)
from .schema import SchemaManager


logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """Configuration for vector indices."""
    ef_construction: int = 200
    ef_search: int = 100
    m: int = 16
    nlist: int = 1024
    nprobe: int = 64
    metric: DistanceMetric = DistanceMetric.COSINE


class VectorIndex:
    """Manages vector indices using FAISS and custom algorithms."""
    
    def __init__(self, schema: VectorSchema, config: IndexConfig):
        self.schema = schema
        self.config = config
        self.index = None
        self.id_mapping: Dict[int, VectorId] = {}
        self.reverse_mapping: Dict[VectorId, int] = {}
        self.next_id = 0
        self._lock = threading.RLock()
        
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize the appropriate FAISS index based on schema."""
        d = self.schema.dimensions
        
        if self.schema.index_type == IndexType.FLAT:
            if self.schema.metric == DistanceMetric.COSINE:
                self.index = faiss.IndexFlatIP(d)  # Inner product for normalized vectors
            elif self.schema.metric == DistanceMetric.EUCLIDEAN:
                self.index = faiss.IndexFlatL2(d)
            else:
                self.index = faiss.IndexFlat(d)
                
        elif self.schema.index_type == IndexType.HNSW:
            if self.schema.metric == DistanceMetric.COSINE:
                self.index = faiss.IndexHNSWFlat(d, self.config.m)
                self.index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                self.index = faiss.IndexHNSWFlat(d, self.config.m)
            
            self.index.hnsw.efConstruction = self.config.ef_construction
            
        elif self.schema.index_type == IndexType.IVF:
            quantizer = faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, self.config.nlist)
            
        else:
            raise IndexError(
                "initialize", 
                self.schema.index_type, 
                f"Unsupported index type: {self.schema.index_type}"
            )
    
    def add_vectors(self, vectors: List[Vector]) -> List[int]:
        """Add vectors to the index."""
        with self._lock:
            if not vectors:
                return []
            
            # Prepare data
            vector_data = np.array([v.data for v in vectors], dtype=np.float32)
            
            # Normalize for cosine similarity
            if self.schema.metric == DistanceMetric.COSINE:
                norms = np.linalg.norm(vector_data, axis=1, keepdims=True)
                vector_data = vector_data / np.maximum(norms, 1e-12)
            
            # Train index if needed
            if not self.index.is_trained:
                self.index.train(vector_data)
            
            # Add to index - different approach for different index types
            start_id = self.next_id
            faiss_ids = list(range(start_id, start_id + len(vectors)))
            
            if hasattr(self.index, 'add_with_ids') and self.schema.index_type in [IndexType.FLAT, IndexType.IVF]:
                # Use add_with_ids for indices that support it
                self.index.add_with_ids(vector_data, np.array(faiss_ids, dtype=np.int64))
            else:
                # Use regular add for HNSW and other indices
                self.index.add(vector_data)
            
            # Update mappings
            for i, vector in enumerate(vectors):
                faiss_id = faiss_ids[i]
                self.id_mapping[faiss_id] = vector.id
                self.reverse_mapping[vector.id] = faiss_id
            
            self.next_id += len(vectors)
            return faiss_ids
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar vectors."""
        with self._lock:
            if self.index.ntotal == 0:
                return []
            
            # Prepare query vector
            query_vector = np.array([query.vector], dtype=np.float32)
            
            if self.schema.metric == DistanceMetric.COSINE:
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = query_vector / norm
            
            # Set search parameters
            if self.schema.index_type == IndexType.HNSW:
                faiss.ParameterSpace().set_index_parameter(
                    self.index, "efSearch", query.ef or self.config.ef_search
                )
            elif self.schema.index_type == IndexType.IVF:
                self.index.nprobe = query.nprobe or self.config.nprobe
            
            # Perform search
            k = min(query.k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, k)
            
            # Convert results
            results = []
            for i in range(k):
                if indices[0][i] == -1:  # No more results
                    break
                
                faiss_id = indices[0][i]
                distance = distances[0][i]
                
                if faiss_id in self.id_mapping:
                    vector_id = self.id_mapping[faiss_id]
                    
                    # Convert distance to score
                    if self.schema.metric == DistanceMetric.COSINE:
                        score = float(distance)  # Already similarity for IP
                    else:
                        score = 1.0 / (1.0 + float(distance))  # Convert distance to similarity
                    
                    # Apply threshold filter
                    if query.threshold is None or score >= query.threshold:
                        results.append(SearchResult(
                            id=vector_id,
                            vector=None,  # Will be populated later if requested
                            score=score,
                            metadata={},  # Will be populated later
                            distance=float(distance)
                        ))
            
            return results
    
    def remove_vector(self, vector_id: VectorId) -> bool:
        """Remove a vector from the index."""
        with self._lock:
            if vector_id not in self.reverse_mapping:
                return False
            
            faiss_id = self.reverse_mapping[vector_id]
            
            # FAISS doesn't support efficient removal, so we'll mark as removed
            # and rebuild periodically
            del self.id_mapping[faiss_id]
            del self.reverse_mapping[vector_id]
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self._lock:
            return {
                "total_vectors": self.index.ntotal,
                "index_type": self.schema.index_type,
                "metric": self.schema.metric,
                "dimensions": self.schema.dimensions,
                "memory_usage_bytes": self.index.ntotal * self.schema.dimensions * 4,  # FP32
                "is_trained": self.index.is_trained
            }


class CompressionEngine:
    """Handles vector and metadata compression."""
    
    @staticmethod
    def compress_data(data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using the specified algorithm."""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.LZ4:
            return lz4.frame.compress(data)
        else:
            raise ValidationError("compression_type", compression_type, "Unsupported compression type")
    
    @staticmethod
    def decompress_data(data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using the specified algorithm."""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.LZ4:
            return lz4.frame.decompress(data)
        else:
            raise ValidationError("compression_type", compression_type, "Unsupported compression type")
    
    @staticmethod
    def quantize_vector(vector: np.ndarray, quantization: QuantizationType) -> np.ndarray:
        """Quantize vector to reduce memory usage."""
        if quantization == QuantizationType.NONE:
            return vector.astype(np.float32)
        elif quantization == QuantizationType.FP16:
            return vector.astype(np.float16)
        elif quantization == QuantizationType.INT8:
            # Simple quantization to int8
            min_val, max_val = vector.min(), vector.max()
            if max_val > min_val:
                scale = 127.0 / (max_val - min_val)
                quantized = ((vector - min_val) * scale).astype(np.int8)
                return quantized
            else:
                return np.zeros_like(vector, dtype=np.int8)
        else:
            raise ValidationError("quantization", quantization, "Unsupported quantization type")


class EncryptionEngine:
    """Handles data encryption and decryption."""
    
    def __init__(self, key: Optional[str] = None):
        self.fernet = None
        if key:
            self._derive_key(key.encode())
    
    def _derive_key(self, password: bytes) -> None:
        """Derive encryption key from password."""
        salt = b'toucandb_salt_2023'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password)
        # Use the derived key with base64 encoding for Fernet
        import base64
        key_b64 = base64.urlsafe_b64encode(key)
        self.fernet = Fernet(key_b64)
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data."""
        if not self.fernet:
            return data
        try:
            return self.fernet.encrypt(data)
        except Exception as e:
            raise EncryptionError("encrypt", str(e))
    
    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data."""
        if not self.fernet:
            return data
        try:
            return self.fernet.decrypt(data)
        except Exception as e:
            raise EncryptionError("decrypt", str(e))


class VectorStorage:
    """Manages persistent storage of vectors and metadata."""
    
    def __init__(
        self, 
        storage_path: Path, 
        schema: VectorSchema,
        encryption_key: Optional[str] = None
    ):
        self.storage_path = storage_path
        self.schema = schema
        self.vectors_path = storage_path / "vectors"
        self.metadata_path = storage_path / "metadata"
        
        # Create directories
        self.vectors_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize engines
        self.compression = CompressionEngine()
        self.encryption = EncryptionEngine(encryption_key)
        
        # In-memory cache
        self._vector_cache: Dict[VectorId, Vector] = {}
        self._cache_lock = threading.RLock()
    
    def store_vector(self, vector: Vector) -> None:
        """Store a vector to disk."""
        # Quantize vector
        quantized_data = self.compression.quantize_vector(
            vector.data, self.schema.quantization
        )
        
        # Serialize vector data
        vector_data = {
            'id': vector.id,
            'data': quantized_data.tobytes(),
            'shape': quantized_data.shape,
            'dtype': str(quantized_data.dtype),
            'timestamp': vector.timestamp.isoformat()
        }
        
        # Serialize metadata
        metadata_data = {
            'id': vector.id,
            'metadata': vector.metadata,
            'timestamp': vector.timestamp.isoformat()
        }
        
        # Compress and encrypt
        vector_bytes = msgpack.packb(vector_data)
        metadata_bytes = msgpack.packb(metadata_data)
        
        vector_bytes = self.compression.compress_data(vector_bytes, self.schema.compression)
        metadata_bytes = self.compression.compress_data(metadata_bytes, self.schema.compression)
        
        vector_bytes = self.encryption.encrypt(vector_bytes)
        metadata_bytes = self.encryption.encrypt(metadata_bytes)
        
        # Write to disk
        vector_file = self.vectors_path / f"{vector.id}.vec"
        metadata_file = self.metadata_path / f"{vector.id}.meta"
        
        try:
            with open(vector_file, 'wb') as f:
                f.write(vector_bytes)
            
            with open(metadata_file, 'wb') as f:
                f.write(metadata_bytes)
            
            # Update cache
            with self._cache_lock:
                self._vector_cache[vector.id] = vector
                
        except Exception as e:
            raise StorageError("write", str(vector_file), str(e))
    
    def load_vector(self, vector_id: VectorId) -> Optional[Vector]:
        """Load a vector from disk or cache."""
        # Check cache first
        with self._cache_lock:
            if vector_id in self._vector_cache:
                return self._vector_cache[vector_id]
        
        vector_file = self.vectors_path / f"{vector_id}.vec"
        metadata_file = self.metadata_path / f"{vector_id}.meta"
        
        if not vector_file.exists() or not metadata_file.exists():
            return None
        
        try:
            # Load and decrypt
            with open(vector_file, 'rb') as f:
                vector_bytes = f.read()
            
            with open(metadata_file, 'rb') as f:
                metadata_bytes = f.read()
            
            vector_bytes = self.encryption.decrypt(vector_bytes)
            metadata_bytes = self.encryption.decrypt(metadata_bytes)
            
            # Decompress
            vector_bytes = self.compression.decompress_data(vector_bytes, self.schema.compression)
            metadata_bytes = self.compression.decompress_data(metadata_bytes, self.schema.compression)
            
            # Deserialize
            vector_data = msgpack.unpackb(vector_bytes, raw=False)
            metadata_data = msgpack.unpackb(metadata_bytes, raw=False)
            
            # Reconstruct vector
            data_bytes = vector_data['data']
            shape = vector_data['shape']
            dtype = np.dtype(vector_data['dtype'])
            
            vector_array = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
            
            vector = Vector(
                id=vector_data['id'],
                data=vector_array.astype(np.float32),  # Convert back to float32
                metadata=metadata_data['metadata'],
                timestamp=datetime.fromisoformat(vector_data['timestamp'])
            )
            
            # Update cache
            with self._cache_lock:
                self._vector_cache[vector_id] = vector
            
            return vector
            
        except Exception as e:
            raise StorageError("read", str(vector_file), str(e))
    
    def delete_vector(self, vector_id: VectorId) -> bool:
        """Delete a vector from storage."""
        vector_file = self.vectors_path / f"{vector_id}.vec"
        metadata_file = self.metadata_path / f"{vector_id}.meta"
        
        deleted = False
        
        try:
            if vector_file.exists():
                vector_file.unlink()
                deleted = True
            
            if metadata_file.exists():
                metadata_file.unlink()
                deleted = True
            
            # Remove from cache
            with self._cache_lock:
                self._vector_cache.pop(vector_id, None)
            
            return deleted
            
        except Exception as e:
            raise StorageError("delete", str(vector_file), str(e))
    
    def list_vectors(self) -> List[VectorId]:
        """List all stored vector IDs."""
        vector_ids = []
        for vector_file in self.vectors_path.glob("*.vec"):
            vector_ids.append(vector_file.stem)
        return vector_ids
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = 0
        vector_count = 0
        
        for vector_file in self.vectors_path.glob("*.vec"):
            total_size += vector_file.stat().st_size
            vector_count += 1
        
        for metadata_file in self.metadata_path.glob("*.meta"):
            total_size += metadata_file.stat().st_size
        
        return {
            "total_vectors": vector_count,
            "total_size_bytes": total_size,
            "cache_size": len(self._vector_cache),
            "compression": self.schema.compression,
            "quantization": self.schema.quantization
        }


class VectorCollection:
    """Represents a collection of vectors with a specific schema."""
    
    def __init__(
        self, 
        name: str, 
        schema: VectorSchema, 
        storage_path: Path,
        encryption_key: Optional[str] = None
    ):
        self.name = name
        self.schema = schema
        self.storage_path = storage_path / name
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        index_config = IndexConfig(
            ef_construction=schema.hnsw_ef_construction,
            m=schema.hnsw_m,
            nlist=schema.ivf_nlist,
            metric=schema.metric
        )
        
        self.index = VectorIndex(schema, index_config)
        self.storage = VectorStorage(self.storage_path, schema, encryption_key)
        
        # Statistics
        self._stats = CollectionStats(
            name=name,
            total_vectors=0,
            dimensions=schema.dimensions,
            size_bytes=0,
            index_type=schema.index_type,
            metric=schema.metric,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            avg_search_latency_ms=0.0,
            cache_hit_ratio=0.0,
            compression_ratio=0.0
        )
        
        # Load existing vectors
        self._load_existing_vectors()
    
    def _load_existing_vectors(self) -> None:
        """Load existing vectors into the index."""
        vector_ids = self.storage.list_vectors()
        if not vector_ids:
            return
        
        # Load vectors in batches
        batch_size = 1000
        for i in range(0, len(vector_ids), batch_size):
            batch_ids = vector_ids[i:i + batch_size]
            vectors = []
            
            for vector_id in batch_ids:
                vector = self.storage.load_vector(vector_id)
                if vector:
                    vectors.append(vector)
            
            if vectors:
                self.index.add_vectors(vectors)
                self._stats.total_vectors += len(vectors)
    
    async def insert(self, vectors: List[Dict[str, Any]]) -> OperationResult[List[VectorId]]:
        """Insert vectors into the collection."""
        start_time = time.time()
        
        try:
            # Validate and convert to Vector objects
            vector_objects = []
            for vec_data in vectors:
                if 'id' not in vec_data or 'vector' not in vec_data:
                    return OperationResult.error_result(
                        ErrorCode.VALIDATION_ERROR,
                        "Each vector must have 'id' and 'vector' fields"
                    )
                
                # Validate dimensions
                if len(vec_data['vector']) != self.schema.dimensions:
                    return OperationResult.error_result(
                        ErrorCode.DIMENSION_MISMATCH,
                        f"Expected {self.schema.dimensions} dimensions, got {len(vec_data['vector'])}"
                    )
                
                vector = Vector(
                    id=vec_data['id'],
                    data=np.array(vec_data['vector'], dtype=np.float32),
                    metadata=vec_data.get('metadata', {}),
                    timestamp=datetime.utcnow()
                )
                vector_objects.append(vector)
            
            # Store vectors
            for vector in vector_objects:
                self.storage.store_vector(vector)
            
            # Add to index
            self.index.add_vectors(vector_objects)
            
            # Update statistics
            self._stats.total_vectors += len(vector_objects)
            self._stats.updated_at = datetime.utcnow()
            
            execution_time = (time.time() - start_time) * 1000
            
            return OperationResult.success_result(
                [v.id for v in vector_objects],
                execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return OperationResult.error_result(
                ErrorCode.STORAGE_ERROR,
                str(e),
                execution_time
            )
    
    async def search(self, query: SearchQuery) -> OperationResult[List[SearchResult]]:
        """Search for similar vectors."""
        start_time = time.time()
        
        try:
            # Validate query vector dimensions
            if len(query.vector) != self.schema.dimensions:
                return OperationResult.error_result(
                    ErrorCode.DIMENSION_MISMATCH,
                    f"Query vector has {len(query.vector)} dimensions, expected {self.schema.dimensions}"
                )
            
            # Perform search
            results = self.index.search(query)
            
            # Populate metadata and vectors if requested
            for result in results:
                if query.include_metadata or query.include_vectors:
                    vector = self.storage.load_vector(result.id)
                    if vector:
                        if query.include_metadata:
                            result.metadata = vector.metadata
                        if query.include_vectors:
                            result.vector = vector.data
            
            # Apply metadata filters
            if query.metadata_filter:
                filtered_results = []
                for result in results:
                    if self._matches_filter(result.metadata, query.metadata_filter):
                        filtered_results.append(result)
                results = filtered_results
            
            execution_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self._stats.avg_search_latency_ms = (
                self._stats.avg_search_latency_ms * 0.9 + execution_time * 0.1
            )
            
            return OperationResult.success_result(results, execution_time)
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return OperationResult.error_result(
                ErrorCode.INDEX_ERROR,
                str(e),
                execution_time
            )
    
    def _matches_filter(self, metadata: MetadataDict, filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    async def delete(self, vector_id: VectorId) -> OperationResult[bool]:
        """Delete a vector from the collection."""
        start_time = time.time()
        
        try:
            # Remove from storage
            storage_deleted = self.storage.delete_vector(vector_id)
            
            # Remove from index
            index_deleted = self.index.remove_vector(vector_id)
            
            deleted = storage_deleted or index_deleted
            
            if deleted:
                self._stats.total_vectors -= 1
                self._stats.updated_at = datetime.utcnow()
            
            execution_time = (time.time() - start_time) * 1000
            
            return OperationResult.success_result(deleted, execution_time)
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return OperationResult.error_result(
                ErrorCode.STORAGE_ERROR,
                str(e),
                execution_time
            )
    
    def get_stats(self) -> CollectionStats:
        """Get collection statistics."""
        # Update size from storage
        storage_stats = self.storage.get_storage_stats()
        self._stats.size_bytes = storage_stats["total_size_bytes"]
        self._stats.compression_ratio = storage_stats.get("compression_ratio", 1.0)
        
        return self._stats