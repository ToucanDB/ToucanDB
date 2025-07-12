#!/usr/bin/env python3
"""
Quick ToucanDB test to verify functionality.
"""
import asyncio
import numpy as np
from toucandb import ToucanDB, create_schema, SearchQuery

async def test_basic_functionality():
    print("🦜 Testing ToucanDB Basic Functionality")
    print("=" * 40)
    
    # Create a temporary database
    db = await ToucanDB.create("./test_db", encryption_key="test-key")
    
    try:
        # Create a simple schema
        schema = create_schema("test", 128, "cosine", "hnsw")
        collection = await db.create_collection(schema)
        print(f"✓ Created collection: {collection.name}")
        
        # Insert test vectors
        vectors = []
        for i in range(10):
            vector_data = np.random.rand(128).tolist()
            vectors.append({
                "id": f"test_vec_{i}",
                "vector": vector_data,
                "metadata": {"index": i, "category": "test"}
            })
        
        result = await db.insert_vectors("test", vectors)
        if result.success:
            print(f"✓ Inserted {len(result.data)} vectors")
        else:
            print(f"✗ Insert failed: {result.error_message}")
            return
        
        # Test search
        query = SearchQuery(vector=vectors[0]["vector"], k=3, include_metadata=True)
        search_result = await db.search_vectors("test", query)
        
        print(f"✓ Found {len(search_result.data)} similar vectors")
        print(f"✓ Best match score: {search_result.data[0]['score']:.3f}")
        
        # Get stats
        stats = db.get_collection_stats("test")
        print(f"✓ Collection has {stats.total_vectors} total vectors")
        
        print("\n🎉 All tests passed! ToucanDB is working correctly.")
        
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
