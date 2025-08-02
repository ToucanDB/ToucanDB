#!/usr/bin/env python3
"""
ToucanDB Quick Start Example

This script demonstrates the basic usage of ToucanDB including:
- Database creation
- Collection setup
- Vector insertion
- Similarity search
- Performance monitoring
"""

import asyncio
import numpy as np
import time
from pathlib import Path

from toucandb import ToucanDB, create_schema, SearchQuery


async def basic_example():
    """Basic ToucanDB usage example."""
    print("🦜 ToucanDB Quick Start Example")
    print("=" * 40)

    # 1. Create database
    print("\n1. Creating ToucanDB database...")
    db_path = Path("./example_db")
    db = await ToucanDB.create(db_path, encryption_key="my-secret-key-123")

    print(f"   ✓ Database created at {db_path}")

    try:
        # 2. Create a collection schema
        print("\n2. Creating collection schema...")
        schema = create_schema(
            name="documents",
            dimensions=384,  # Common for sentence transformers
            metric="cosine",
            index_type="hnsw"
        )

        collection = await db.create_collection(schema)
        print(f"   ✓ Collection '{schema.name}' created with {schema.dimensions}D vectors")

        # 3. Prepare sample vectors (simulating document embeddings)
        print("\n3. Generating sample document embeddings...")
        documents = [
            {"title": "Machine Learning Basics", "content": "Introduction to ML concepts"},
            {"title": "Deep Learning Guide", "content": "Neural networks and deep learning"},
            {"title": "AI Ethics", "content": "Responsible AI development"},
            {"title": "Computer Vision", "content": "Image processing with AI"},
            {"title": "Natural Language Processing", "content": "Understanding human language"},
            {"title": "Data Science Tools", "content": "Python, pandas, and analytics"},
            {"title": "Cloud Computing", "content": "Scalable infrastructure"},
            {"title": "Database Systems", "content": "SQL and NoSQL databases"},
            {"title": "Web Development", "content": "Frontend and backend technologies"},
            {"title": "Cybersecurity", "content": "Protecting digital assets"}
        ]

        # Generate random embeddings (in real usage, use actual embedding models)
        vectors = []
        for i, doc in enumerate(documents):
            # Simulate embeddings with some structure
            embedding = np.random.rand(384).astype(np.float32)
            # Add some correlation for demonstration
            if "AI" in doc["title"] or "ML" in doc["title"] or "Deep" in doc["title"]:
                embedding[:50] += 0.5  # Make AI-related docs similar

            vectors.append({
                "id": f"doc_{i}",
                "vector": embedding.tolist(),
                "metadata": {
                    "title": doc["title"],
                    "content": doc["content"],
                    "category": "AI/ML" if any(term in doc["title"] for term in ["AI", "ML", "Deep", "Computer Vision", "NLP"]) else "Tech",
                    "length": len(doc["content"]),
                    "created_at": "2024-01-01"
                }
            })

        print(f"   ✓ Generated {len(vectors)} document embeddings")

        # 4. Insert vectors into collection
        print("\n4. Inserting vectors into collection...")
        start_time = time.time()

        result = await db.insert_vectors("documents", vectors)

        insert_time = time.time() - start_time

        if result.success:
            print(f"   ✓ Inserted {len(result.data)} vectors in {insert_time:.3f}s")
            print(f"   ✓ Average: {len(result.data)/insert_time:.1f} vectors/second")
        else:
            print(f"   ✗ Insertion failed: {result.error_message}")
            return

        # 5. Perform similarity searches
        print("\n5. Performing similarity searches...")

        # Search for AI-related content
        query_vector = vectors[0]["vector"]  # Use first doc (ML Basics) as query

        search_queries = [
            ("Basic search (k=3)", SearchQuery(vector=query_vector, k=3)),
            ("With metadata", SearchQuery(vector=query_vector, k=5, include_metadata=True)),
            ("Filtered by category", SearchQuery(
                vector=query_vector,
                k=5,
                include_metadata=True,
                metadata_filter={"category": "AI/ML"}
            )),
            ("High threshold", SearchQuery(
                vector=query_vector,
                k=10,
                include_metadata=True,
                threshold=0.7
            ))
        ]

        for query_name, query in search_queries:
            print(f"\n   🔍 {query_name}:")

            start_time = time.time()
            result = await db.search_vectors("documents", query)
            search_time = time.time() - start_time

            if result.success:
                print(f"      ✓ Found {len(result.data)} results in {search_time*1000:.1f}ms")

                for i, hit in enumerate(result.data[:3]):  # Show top 3
                    score_bar = "█" * int(hit["score"] * 10) + "░" * (10 - int(hit["score"] * 10))
                    print(f"      {i+1}. {score_bar} {hit['score']:.3f}")

                    if "metadata" in hit:
                        print(f"         Title: {hit['metadata']['title']}")
                        print(f"         Category: {hit['metadata']['category']}")
            else:
                print(f"      ✗ Search failed: {result.error_message}")

        # 6. Collection statistics
        print("\n6. Collection statistics:")
        stats = db.get_collection_stats("documents")

        print(f"   📊 Total vectors: {stats.total_vectors}")
        print(f"   📊 Dimensions: {stats.dimensions}")
        print(f"   📊 Storage size: {stats.size_bytes:,} bytes")
        print(f"   📊 Index type: {stats.index_type}")
        print(f"   📊 Distance metric: {stats.metric}")
        print(f"   📊 Avg search latency: {stats.avg_search_latency_ms:.2f}ms")

        # 7. Database information
        print("\n7. Database overview:")
        db_info = db.get_database_info()

        print(f"   🗄️  Total collections: {db_info['total_collections']}")
        print(f"   🗄️  Total vectors: {db_info['total_vectors']:,}")
        print(f"   🗄️  Storage path: {db_info['storage_path']}")
        print(f"   🗄️  Version: {db_info['version']}")

        # 8. Demonstrate vector deletion
        print("\n8. Demonstrating vector deletion...")
        delete_result = await db.delete_vector("documents", "doc_9")

        if delete_result.success:
            new_stats = db.get_collection_stats("documents")
            print(f"   ✓ Deleted vector 'doc_9'")
            print(f"   ✓ Vector count: {stats.total_vectors} → {new_stats.total_vectors}")

        print("\n" + "=" * 40)
        print("🎉 ToucanDB demo completed successfully!")
        print("\nKey features demonstrated:")
        print("  • Secure vector storage with encryption")
        print("  • Fast similarity search with HNSW indexing")
        print("  • Metadata filtering and rich queries")
        print("  • Real-time performance monitoring")
        print("  • Type-safe async operations")

    finally:
        # 9. Clean up
        await db.close()
        print(f"\n🧹 Database closed. Demo data saved at {db_path}")


async def advanced_example():
    """Advanced ToucanDB features demonstration."""
    print("\n" + "=" * 50)
    print("🚀 Advanced ToucanDB Features")
    print("=" * 50)

    db_path = Path("./advanced_db")
    db = await ToucanDB.create(db_path, encryption_key="advanced-key-456")

    try:
        # Multiple collections with different configurations
        print("\n1. Creating multiple specialized collections...")

        schemas = [
            create_schema("images", 2048, "euclidean", "ivf"),      # Image embeddings
            create_schema("text", 768, "cosine", "hnsw"),           # Text embeddings
            create_schema("audio", 512, "dot_product", "flat")      # Audio embeddings
        ]

        collections = {}
        for schema in schemas:
            collections[schema.name] = await db.create_collection(schema)
            print(f"   ✓ Created '{schema.name}' collection ({schema.dimensions}D, {schema.metric})")

        # Batch operations
        print("\n2. Demonstrating batch operations...")

        # Generate and insert vectors for each collection
        for collection_name, collection in collections.items():
            dims = collection.schema.dimensions
            batch_vectors = []

            for i in range(50):
                vector_data = np.random.rand(dims).astype(np.float32)
                batch_vectors.append({
                    "id": f"{collection_name}_{i}",
                    "vector": vector_data.tolist(),
                    "metadata": {
                        "type": collection_name,
                        "batch": i // 10,
                        "quality": np.random.choice(["high", "medium", "low"]),
                        "timestamp": f"2024-01-{(i % 28) + 1:02d}"
                    }
                })

            start_time = time.time()
            result = await db.insert_vectors(collection_name, batch_vectors, batch_size=25)
            batch_time = time.time() - start_time

            print(f"   ✓ {collection_name}: {len(batch_vectors)} vectors in {batch_time:.3f}s")

        # Complex queries
        print("\n3. Complex multi-filter queries...")

        # Query text collection with multiple filters
        query_vector = np.random.rand(768).tolist()
        complex_query = SearchQuery(
            vector=query_vector,
            k=10,
            include_metadata=True,
            include_vectors=False,  # Don't return vectors to save bandwidth
            metadata_filter={"quality": "high"},
            threshold=0.3
        )

        result = await db.search_vectors("text", complex_query)
        if result.success:
            print(f"   ✓ Found {len(result.data)} high-quality text vectors")

            # Group results by batch
            batches = {}
            for hit in result.data:
                batch = hit["metadata"]["batch"]
                if batch not in batches:
                    batches[batch] = []
                batches[batch].append(hit)

            print(f"   ✓ Results span {len(batches)} different batches")

        # Performance comparison
        print("\n4. Performance comparison across collections...")

        for collection_name in collections.keys():
            collection = collections[collection_name]
            dims = collection.schema.dimensions

            # Generate query
            query_vector = np.random.rand(dims).tolist()
            query = SearchQuery(vector=query_vector, k=20)

            # Time multiple searches
            times = []
            for _ in range(5):
                start = time.time()
                await db.search_vectors(collection_name, query)
                times.append((time.time() - start) * 1000)

            avg_time = np.mean(times)
            std_time = np.std(times)

            print(f"   📈 {collection_name:8s}: {avg_time:6.2f}ms ± {std_time:5.2f}ms")

        # Database statistics
        print("\n5. Comprehensive database statistics...")

        total_vectors = 0
        total_size = 0

        for collection_name in collections.keys():
            stats = db.get_collection_stats(collection_name)
            total_vectors += stats.total_vectors
            total_size += stats.size_bytes

            print(f"   📊 {collection_name}:")
            print(f"      Vectors: {stats.total_vectors:,}")
            print(f"      Size: {stats.size_bytes:,} bytes")
            print(f"      Avg latency: {stats.avg_search_latency_ms:.2f}ms")

        print(f"\n   🎯 Total: {total_vectors:,} vectors, {total_size:,} bytes")

        # Cleanup demonstration
        print("\n6. Collection management...")

        # Drop a collection
        dropped = await db.drop_collection("audio")
        if dropped:
            print("   ✓ Dropped 'audio' collection")
            print(f"   ✓ Remaining collections: {db.list_collections()}")

        print("\n" + "=" * 50)
        print("🌟 Advanced features completed!")
        print("\nAdvanced capabilities shown:")
        print("  • Multiple collection types and configurations")
        print("  • Batch processing for high throughput")
        print("  • Complex metadata filtering")
        print("  • Performance monitoring and optimization")
        print("  • Dynamic collection management")

    finally:
        await db.close()
        print(f"\n🧹 Advanced demo completed. Data at {db_path}")


async def main():
    """Run all examples."""
    try:
        await basic_example()
        await advanced_example()

        print("\n" + "🎊" * 20)
        print("All ToucanDB examples completed successfully!")
        print("Check the generated database directories for persisted data.")
        print("🎊" * 20)

    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
