"""
ToucanDB - A Secure, Efficient ML-First Vector Database Engine

This module provides the main interface for ToucanDB, allowing users to
create, manage, and query vector collections with state-of-the-art performance
and security features.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime

from .types import (
    VectorSchema, SearchQuery, InsertRequest, DatabaseConfig,
    OperationResult, CollectionStats, ErrorCode
)
from .exceptions import (
    ToucanDBException, CollectionNotFoundError, InvalidSchemaError,
    StorageError, ConfigurationError
)
from .schema import SchemaManager
from .vector_engine import VectorCollection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__author__ = "Pierre-Henry Soria"
__email__ = "pierre@ph7.me"
__license__ = "MIT"
__description__ = "A secure, efficient ML-first vector database engine"

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
    "InvalidSchemaError"
]


class ToucanDB:
    """
    Main ToucanDB database interface.
    
    Provides a high-level API for creating and managing vector collections,
    with built-in security, compression, and performance optimizations.
    """
    
    def __init__(self, storage_path: Union[str, Path], config: Optional[DatabaseConfig] = None):
        """
        Initialize ToucanDB instance.
        
        Args:
            storage_path: Path to database storage directory
            config: Database configuration (optional)
        """
        self.storage_path = Path(storage_path)
        self.config = config or self._default_config()
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.schema_manager = SchemaManager(self.storage_path)
        self.collections: Dict[str, VectorCollection] = {}
        
        # Database metadata
        self.created_at = datetime.utcnow()
        self.last_backup = None
        
        # Load existing collections
        self._load_collections()
        
        logger.info(f"ToucanDB initialized at {self.storage_path}")
    
    @classmethod
    async def create(
        cls, 
        storage_path: Union[str, Path], 
        config: Optional[DatabaseConfig] = None,
        encryption_key: Optional[str] = None
    ) -> 'ToucanDB':
        """
        Create a new ToucanDB instance asynchronously.
        
        Args:
            storage_path: Path to database storage directory
            config: Database configuration
            encryption_key: Encryption key for data security
            
        Returns:
            ToucanDB instance
        """
        if config is None:
            config = cls._default_config()
            if encryption_key:
                config.storage.encryption_key = encryption_key
        
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
        
        for collection_name in collection_names:
            schema = self.schema_manager.get_schema(collection_name)
            if schema:
                try:
                    collection = VectorCollection(
                        name=collection_name,
                        schema=schema,
                        storage_path=self.storage_path,
                        encryption_key=self.config.storage.encryption_key
                    )
                    self.collections[collection_name] = collection
                    logger.info(f"Loaded collection: {collection_name}")
                except Exception as e:
                    logger.error(f"Failed to load collection {collection_name}: {e}")
    
    async def create_collection(
        self, 
        schema: VectorSchema,
        overwrite: bool = False
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
            # Create schema
            schema_version = self.schema_manager.create_schema(
                schema.name, schema, overwrite
            )
            
            # Create collection
            collection = VectorCollection(
                name=schema.name,
                schema=schema,
                storage_path=self.storage_path,
                encryption_key=self.config.storage.encryption_key
            )
            
            self.collections[schema.name] = collection
            
            logger.info(f"Created collection: {schema.name}")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create collection {schema.name}: {e}")
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
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        return list(self.collections.keys())
    
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
            # Remove from memory
            del self.collections[name]
            
            # Mark schema as inactive
            self.schema_manager.delete_schema(name)
            
            # TODO: Clean up storage files
            logger.info(f"Dropped collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop collection {name}: {e}")
            raise StorageError("drop_collection", name, str(e))
    
    async def insert_vectors(
        self, 
        collection_name: str, 
        vectors: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> OperationResult[List[str]]:
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
        
        # Process in batches for large datasets
        all_ids = []
        total_time = 0.0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            result = await collection.insert(batch)
            
            if not result.success:
                return result
            
            all_ids.extend(result.data)
            total_time += result.execution_time_ms
        
        return OperationResult.success_result(all_ids, total_time)
    
    async def search_vectors(
        self, 
        collection_name: str, 
        query: SearchQuery
    ) -> OperationResult[List[Dict[str, Any]]]:
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
            # Convert SearchResult objects to dictionaries
            search_results = []
            for sr in result.data:
                result_dict = {
                    "id": sr.id,
                    "score": sr.score,
                    "distance": sr.distance
                }
                
                if query.include_metadata:
                    result_dict["metadata"] = sr.metadata
                
                if query.include_vectors:
                    result_dict["vector"] = sr.vector.tolist() if sr.vector is not None else None
                
                search_results.append(result_dict)
            
            return OperationResult.success_result(search_results, result.execution_time_ms)
        
        return result
    
    async def delete_vector(
        self, 
        collection_name: str, 
        vector_id: str
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
    
    def get_database_info(self) -> Dict[str, Any]:
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
                "metric": stats.metric
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
            "config": self.config.dict(),
            "collections": collection_info
        }
    
    async def backup(self, backup_path: Union[str, Path]) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for backup files
            
        Returns:
            True if backup was successful
        """
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export schemas
            schema_backup = backup_path / "schemas.json"
            self.schema_manager.export_schemas(schema_backup)
            
            # TODO: Backup vector data and indices
            
            self.last_backup = datetime.utcnow()
            logger.info(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    async def close(self) -> None:
        """
        Close the database and clean up resources.
        """
        logger.info("Closing ToucanDB...")
        
        # Close collections
        for collection in self.collections.values():
            # TODO: Implement collection cleanup
            pass
        
        self.collections.clear()
        logger.info("ToucanDB closed successfully")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience functions
async def create_database(
    storage_path: Union[str, Path],
    encryption_key: Optional[str] = None
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
    **kwargs
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
    from .types import DistanceMetric, IndexType
    
    return VectorSchema(
        name=name,
        dimensions=dimensions,
        metric=DistanceMetric(metric),
        index_type=IndexType(index_type),
        **kwargs
    )