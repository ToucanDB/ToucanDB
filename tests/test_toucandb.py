"""
ToucanDB Test Suite

Comprehensive tests for ToucanDB functionality including:
- Database creation and management
- Collection operations
- Vector insertion and search
- Schema validation
- Error handling
"""

import pytest
import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any

from toucandb import ToucanDB, VectorSchema, SearchQuery, create_schema
from toucandb.types import DistanceMetric, IndexType
from toucandb.exceptions import CollectionNotFoundError, DimensionMismatchError


class TestToucanDB:
    """Test ToucanDB main functionality."""
    
    @pytest.fixture
    async def temp_db(self):
        """Create a temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        db = await ToucanDB.create(temp_dir, encryption_key="test-key-123")
        yield db
        await db.close()
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_schema(self):
        """Create a sample schema for testing."""
        return create_schema(
            name="test_collection",
            dimensions=128,
            metric="cosine",
            index_type="hnsw"
        )
    
    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        import numpy as np
        
        vectors = []
        for i in range(100):
            vector_data = np.random.rand(128).tolist()
            vectors.append({
                "id": f"vec_{i}",
                "vector": vector_data,
                "metadata": {
                    "category": "test",
                    "index": i,
                    "group": "A" if i % 2 == 0 else "B"
                }
            })
        return vectors
    
    async def test_database_creation(self, temp_db):
        """Test database creation and basic info."""
        assert temp_db is not None
        
        info = temp_db.get_database_info()
        assert info["version"] == "0.1.0"
        assert info["total_collections"] == 0
        assert info["total_vectors"] == 0
    
    async def test_collection_creation(self, temp_db, sample_schema):
        """Test collection creation and retrieval."""
        # Create collection
        collection = await temp_db.create_collection(sample_schema)
        assert collection.name == "test_collection"
        assert collection.schema.dimensions == 128
        
        # Verify collection exists
        assert "test_collection" in temp_db.list_collections()
        
        # Get collection
        retrieved = temp_db.get_collection("test_collection")
        assert retrieved.name == collection.name
    
    async def test_collection_not_found(self, temp_db):
        """Test error handling for non-existent collections."""
        with pytest.raises(CollectionNotFoundError):
            temp_db.get_collection("nonexistent")
    
    async def test_vector_insertion(self, temp_db, sample_schema, sample_vectors):
        """Test vector insertion into collections."""
        # Create collection
        await temp_db.create_collection(sample_schema)
        
        # Insert vectors
        result = await temp_db.insert_vectors("test_collection", sample_vectors[:10])
        
        assert result.success
        assert len(result.data) == 10
        assert result.execution_time_ms > 0
        
        # Check collection stats
        stats = temp_db.get_collection_stats("test_collection")
        assert stats.total_vectors == 10
    
    async def test_vector_search(self, temp_db, sample_schema, sample_vectors):
        """Test vector similarity search."""
        # Create collection and insert vectors
        await temp_db.create_collection(sample_schema)
        await temp_db.insert_vectors("test_collection", sample_vectors[:50])
        
        # Perform search
        query_vector = sample_vectors[0]["vector"]
        query = SearchQuery(
            vector=query_vector,
            k=5,
            include_metadata=True,
            threshold=0.5
        )
        
        result = await temp_db.search_vectors("test_collection", query)
        
        assert result.success
        assert len(result.data) <= 5
        assert all("score" in r for r in result.data)
        assert all("metadata" in r for r in result.data)
    
    async def test_dimension_validation(self, temp_db, sample_schema):
        """Test vector dimension validation."""
        await temp_db.create_collection(sample_schema)
        
        # Try to insert vector with wrong dimensions
        wrong_vector = {
            "id": "wrong_dim",
            "vector": [1.0, 2.0],  # Only 2 dimensions instead of 128
            "metadata": {}
        }
        
        result = await temp_db.insert_vectors("test_collection", [wrong_vector])
        assert not result.success
        assert result.error_code == "DIMENSION_MISMATCH"
    
    async def test_metadata_filtering(self, temp_db, sample_schema, sample_vectors):
        """Test metadata-based filtering during search."""
        # Create collection and insert vectors
        await temp_db.create_collection(sample_schema)
        await temp_db.insert_vectors("test_collection", sample_vectors[:20])
        
        # Search with metadata filter
        query = SearchQuery(
            vector=sample_vectors[0]["vector"],
            k=10,
            include_metadata=True,
            metadata_filter={"group": "A"}
        )
        
        result = await temp_db.search_vectors("test_collection", query)
        
        assert result.success
        # All results should have group="A"
        for r in result.data:
            assert r["metadata"]["group"] == "A"
    
    async def test_vector_deletion(self, temp_db, sample_schema, sample_vectors):
        """Test vector deletion."""
        # Create collection and insert vectors
        await temp_db.create_collection(sample_schema)
        await temp_db.insert_vectors("test_collection", sample_vectors[:10])
        
        # Delete a vector
        result = await temp_db.delete_vector("test_collection", "vec_0")
        assert result.success
        assert result.data is True
        
        # Verify vector count decreased
        stats = temp_db.get_collection_stats("test_collection")
        assert stats.total_vectors == 9
    
    async def test_collection_drop(self, temp_db, sample_schema):
        """Test collection deletion."""
        # Create collection
        await temp_db.create_collection(sample_schema)
        assert "test_collection" in temp_db.list_collections()
        
        # Drop collection
        result = await temp_db.drop_collection("test_collection")
        assert result is True
        assert "test_collection" not in temp_db.list_collections()
    
    async def test_batch_operations(self, temp_db, sample_schema, sample_vectors):
        """Test large batch insertions."""
        await temp_db.create_collection(sample_schema)
        
        # Insert large batch
        result = await temp_db.insert_vectors("test_collection", sample_vectors)
        
        assert result.success
        assert len(result.data) == len(sample_vectors)
        
        # Verify all vectors were inserted
        stats = temp_db.get_collection_stats("test_collection")
        assert stats.total_vectors == len(sample_vectors)
    
    async def test_concurrent_operations(self, temp_db, sample_schema, sample_vectors):
        """Test concurrent vector operations."""
        await temp_db.create_collection(sample_schema)
        
        # Split vectors into batches for concurrent insertion
        batch1 = sample_vectors[:30]
        batch2 = sample_vectors[30:60]
        batch3 = sample_vectors[60:]
        
        # Insert batches concurrently
        tasks = [
            temp_db.insert_vectors("test_collection", batch1),
            temp_db.insert_vectors("test_collection", batch2),
            temp_db.insert_vectors("test_collection", batch3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        assert all(r.success for r in results)
        
        # Total vectors should be correct
        stats = temp_db.get_collection_stats("test_collection")
        assert stats.total_vectors == len(sample_vectors)


class TestVectorSchema:
    """Test vector schema functionality."""
    
    def test_schema_creation(self):
        """Test basic schema creation."""
        schema = create_schema(
            name="test",
            dimensions=512,
            metric="euclidean",
            index_type="ivf"
        )
        
        assert schema.name == "test"
        assert schema.dimensions == 512
        assert schema.metric == DistanceMetric.EUCLIDEAN
        assert schema.index_type == IndexType.IVF
    
    def test_schema_validation(self):
        """Test schema validation."""
        # Valid schema
        schema = VectorSchema(
            name="valid",
            dimensions=128,
            metric=DistanceMetric.COSINE,
            index_type=IndexType.HNSW
        )
        assert schema.dimensions == 128
        
        # Invalid dimensions should raise validation error
        with pytest.raises(ValueError):
            VectorSchema(
                name="invalid",
                dimensions=-1,  # Invalid
                metric=DistanceMetric.COSINE,
                index_type=IndexType.HNSW
            )


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    async def temp_db(self):
        """Create a temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        db = await ToucanDB.create(temp_dir)
        yield db
        await db.close()
        shutil.rmtree(temp_dir)
    
    async def test_empty_vector_list(self, temp_db):
        """Test handling of empty vector list."""
        schema = create_schema("test", 128)
        await temp_db.create_collection(schema)
        
        result = await temp_db.insert_vectors("test", [])
        assert result.success
        assert len(result.data) == 0
    
    async def test_invalid_search_dimensions(self, temp_db):
        """Test search with wrong vector dimensions."""
        schema = create_schema("test", 128)
        await temp_db.create_collection(schema)
        
        # Search with wrong dimensions
        query = SearchQuery(vector=[1.0, 2.0], k=5)  # Only 2 dims
        result = await temp_db.search_vectors("test", query)
        
        assert not result.success
        assert result.error_code == "DIMENSION_MISMATCH"
    
    async def test_search_empty_collection(self, temp_db):
        """Test search on empty collection."""
        schema = create_schema("test", 128)
        await temp_db.create_collection(schema)
        
        # Search empty collection
        query = SearchQuery(vector=[0.0] * 128, k=5)
        result = await temp_db.search_vectors("test", query)
        
        assert result.success
        assert len(result.data) == 0


class TestPerformance:
    """Performance and scalability tests."""
    
    @pytest.fixture
    async def temp_db(self):
        """Create a temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        db = await ToucanDB.create(temp_dir)
        yield db
        await db.close()
        shutil.rmtree(temp_dir)
    
    async def test_large_vector_insertion(self, temp_db):
        """Test insertion of large number of vectors."""
        import numpy as np
        
        schema = create_schema("large_test", 256)
        await temp_db.create_collection(schema)
        
        # Generate many vectors
        num_vectors = 1000
        vectors = []
        for i in range(num_vectors):
            vector_data = np.random.rand(256).tolist()
            vectors.append({
                "id": f"large_vec_{i}",
                "vector": vector_data,
                "metadata": {"batch": i // 100}
            })
        
        # Insert in batches
        result = await temp_db.insert_vectors("large_test", vectors, batch_size=100)
        
        assert result.success
        assert len(result.data) == num_vectors
        assert result.execution_time_ms > 0
        
        # Verify final count
        stats = temp_db.get_collection_stats("large_test")
        assert stats.total_vectors == num_vectors
    
    async def test_search_performance(self, temp_db):
        """Test search performance with various parameters."""
        import numpy as np
        
        schema = create_schema("perf_test", 128)
        await temp_db.create_collection(schema)
        
        # Insert test vectors
        vectors = []
        for i in range(500):
            vector_data = np.random.rand(128).tolist()
            vectors.append({
                "id": f"perf_vec_{i}",
                "vector": vector_data,
                "metadata": {"group": i % 10}
            })
        
        await temp_db.insert_vectors("perf_test", vectors)
        
        # Test various search parameters
        query_vector = np.random.rand(128).tolist()
        
        # Small k
        query1 = SearchQuery(vector=query_vector, k=5)
        result1 = await temp_db.search_vectors("perf_test", query1)
        assert result1.success
        
        # Large k
        query2 = SearchQuery(vector=query_vector, k=50)
        result2 = await temp_db.search_vectors("perf_test", query2)
        assert result2.success
        
        # With metadata filter
        query3 = SearchQuery(
            vector=query_vector, 
            k=20, 
            include_metadata=True,
            metadata_filter={"group": 5}
        )
        result3 = await temp_db.search_vectors("perf_test", query3)
        assert result3.success


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
