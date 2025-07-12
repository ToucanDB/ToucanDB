#!/usr/bin/env python3
"""
Semantic Search with ToucanDB

This example demonstrates building a semantic search system using
ToucanDB with real sentence embeddings for document retrieval.
"""

import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from toucandb import ToucanDB, create_schema, SearchQuery


# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    {
        "id": "ai_intro",
        "title": "Introduction to Artificial Intelligence",
        "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.",
        "category": "AI/ML",
        "tags": ["artificial intelligence", "machine learning", "computer science"]
    },
    {
        "id": "ml_basics", 
        "title": "Machine Learning Fundamentals",
        "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models that enable computers to improve their performance on tasks through experience.",
        "category": "AI/ML",
        "tags": ["machine learning", "algorithms", "statistics", "data science"]
    },
    {
        "id": "deep_learning",
        "title": "Deep Learning and Neural Networks", 
        "content": "Deep learning is a subset of machine learning based on artificial neural networks. It uses multiple layers to progressively extract higher-level features from raw input.",
        "category": "AI/ML",
        "tags": ["deep learning", "neural networks", "feature extraction"]
    },
    {
        "id": "python_programming",
        "title": "Python Programming for Beginners",
        "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, and automation.",
        "category": "Programming",
        "tags": ["python", "programming", "web development", "data science"]
    },
    {
        "id": "web_development",
        "title": "Modern Web Development",
        "content": "Web development involves creating websites and web applications. Modern frameworks like React, Vue, and Angular have revolutionized frontend development.",
        "category": "Programming", 
        "tags": ["web development", "javascript", "react", "frontend"]
    },
    {
        "id": "database_design",
        "title": "Database Design Principles",
        "content": "Database design is the process of creating a detailed data model for a database. Good design ensures data integrity, reduces redundancy, and improves performance.",
        "category": "Database",
        "tags": ["database", "data modeling", "sql", "design patterns"]
    },
    {
        "id": "cloud_computing",
        "title": "Cloud Computing Fundamentals",
        "content": "Cloud computing delivers computing services over the internet, including servers, storage, databases, and software, enabling scalable and cost-effective solutions.",
        "category": "Infrastructure",
        "tags": ["cloud computing", "aws", "azure", "scalability"]
    },
    {
        "id": "cybersecurity",
        "title": "Cybersecurity Best Practices",
        "content": "Cybersecurity involves protecting digital systems, networks, and data from cyber threats. Key practices include encryption, access control, and regular security audits.",
        "category": "Security",
        "tags": ["cybersecurity", "encryption", "security", "data protection"]
    },
    {
        "id": "data_science",
        "title": "Data Science and Analytics",
        "content": "Data science combines statistics, mathematics, and computer science to extract insights from data. It involves data collection, cleaning, analysis, and visualization.",
        "category": "Data Science",
        "tags": ["data science", "analytics", "statistics", "visualization"]
    },
    {
        "id": "blockchain",
        "title": "Blockchain Technology Explained",
        "content": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records, called blocks, which are linked and secured using cryptography.",
        "category": "Technology",
        "tags": ["blockchain", "cryptocurrency", "distributed systems", "cryptography"]
    }
]


def simulate_sentence_embeddings(text: str, dimensions: int = 384) -> List[float]:
    """
    Simulate sentence embeddings based on text content.
    In a real application, you would use actual embedding models like:
    - sentence-transformers
    - OpenAI embeddings
    - Google Universal Sentence Encoder
    """
    # Create a simple hash-based embedding simulation
    import hashlib
    
    # Normalize text
    text_lower = text.lower().strip()
    
    # Create base embedding from hash
    hash_obj = hashlib.md5(text_lower.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    
    # Generate pseudo-random but deterministic embedding
    np.random.seed(hash_int % (2**32))
    embedding = np.random.normal(0, 1, dimensions)
    
    # Add semantic structure based on keywords
    keyword_weights = {
        'artificial intelligence': [1.0] * 50 + [0.0] * (dimensions - 50),
        'machine learning': [0.8] * 40 + [0.2] * 10 + [0.0] * (dimensions - 50),
        'deep learning': [0.6] * 30 + [0.4] * 20 + [0.0] * (dimensions - 50),
        'programming': [0.0] * 50 + [1.0] * 40 + [0.0] * (dimensions - 90),
        'python': [0.0] * 50 + [0.8] * 35 + [0.2] * 5 + [0.0] * (dimensions - 90),
        'web development': [0.0] * 90 + [1.0] * 30 + [0.0] * (dimensions - 120),
        'database': [0.0] * 120 + [1.0] * 25 + [0.0] * (dimensions - 145),
        'security': [0.0] * 145 + [1.0] * 20 + [0.0] * (dimensions - 165),
        'data science': [0.0] * 165 + [1.0] * 35 + [0.0] * (dimensions - 200),
    }
    
    # Apply keyword weights
    for keyword, weights in keyword_weights.items():
        if keyword in text_lower:
            weight_array = np.array(weights[:dimensions])
            embedding += weight_array * 0.3
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding.tolist()


async def build_semantic_search():
    """Build a semantic search system with ToucanDB."""
    print("🔍 Building Semantic Search with ToucanDB")
    print("=" * 50)
    
    # Create database
    db_path = Path("./semantic_search_db")
    db = await ToucanDB.create(db_path, encryption_key="semantic-search-key")
    
    try:
        # Create collection for document embeddings
        print("\n1. Setting up document collection...")
        schema = create_schema(
            name="documents",
            dimensions=384,
            metric="cosine",  # Best for sentence embeddings
            index_type="hnsw"
        )
        
        collection = await db.create_collection(schema)
        print(f"   ✓ Created collection with {schema.dimensions}D embeddings")
        
        # Generate embeddings for documents
        print("\n2. Generating document embeddings...")
        vectors = []
        
        for doc in SAMPLE_DOCUMENTS:
            # Combine title and content for embedding
            full_text = f"{doc['title']} {doc['content']}"
            embedding = simulate_sentence_embeddings(full_text, 384)
            
            vectors.append({
                "id": doc["id"],
                "vector": embedding,
                "metadata": {
                    "title": doc["title"],
                    "content": doc["content"],
                    "category": doc["category"],
                    "tags": doc["tags"],
                    "content_length": len(doc["content"]),
                    "word_count": len(doc["content"].split())
                }
            })
        
        print(f"   ✓ Generated embeddings for {len(vectors)} documents")
        
        # Insert document embeddings
        print("\n3. Indexing documents...")
        result = await db.insert_vectors("documents", vectors)
        
        if result.success:
            print(f"   ✓ Indexed {len(result.data)} documents")
        else:
            print(f"   ✗ Indexing failed: {result.error_message}")
            return
        
        # Semantic search examples
        print("\n4. Performing semantic searches...")
        
        search_queries = [
            "What is artificial intelligence?",
            "How to learn programming?", 
            "Database management systems",
            "Web application security",
            "Data analysis and visualization",
            "Blockchain and cryptocurrency"
        ]
        
        for i, query_text in enumerate(search_queries, 1):
            print(f"\n   Query {i}: '{query_text}'")
            print("   " + "-" * 40)
            
            # Generate query embedding
            query_embedding = simulate_sentence_embeddings(query_text, 384)
            
            # Search for similar documents
            search_query = SearchQuery(
                vector=query_embedding,
                k=3,
                include_metadata=True,
                threshold=0.1  # Low threshold to get diverse results
            )
            
            search_result = await db.search_vectors("documents", search_query)
            
            if search_result.success:
                print(f"   Found {len(search_result.data)} relevant documents:")
                
                for j, hit in enumerate(search_result.data, 1):
                    score_percent = hit["score"] * 100
                    title = hit["metadata"]["title"]
                    category = hit["metadata"]["category"]
                    
                    # Visual similarity bar
                    bar_length = int(score_percent / 10)
                    similarity_bar = "█" * bar_length + "░" * (10 - bar_length)
                    
                    print(f"   {j}. {similarity_bar} {score_percent:5.1f}% - {title}")
                    print(f"      Category: {category}")
                    
                    # Show snippet of content
                    content = hit["metadata"]["content"]
                    snippet = content[:100] + "..." if len(content) > 100 else content
                    print(f"      Preview: {snippet}")
                    print()
            else:
                print(f"   ✗ Search failed: {search_result.error_message}")
        
        # Category-based search
        print("\n5. Category-filtered searches...")
        
        categories = ["AI/ML", "Programming", "Database"]
        query_text = "machine learning algorithms"
        query_embedding = simulate_sentence_embeddings(query_text, 384)
        
        for category in categories:
            print(f"\n   Searching in '{category}' category:")
            
            filtered_query = SearchQuery(
                vector=query_embedding,
                k=5,
                include_metadata=True,
                metadata_filter={"category": category}
            )
            
            result = await db.search_vectors("documents", filtered_query)
            
            if result.success and result.data:
                best_match = result.data[0]
                print(f"   Best match: {best_match['metadata']['title']}")
                print(f"   Similarity: {best_match['score']*100:.1f}%")
            else:
                print("   No matches found in this category")
        
        # Advanced search with multiple filters
        print("\n6. Advanced multi-filter search...")
        
        advanced_query = SearchQuery(
            vector=query_embedding,
            k=10,
            include_metadata=True,
            metadata_filter={"category": "AI/ML"},
            threshold=0.2
        )
        
        result = await db.search_vectors("documents", advanced_query)
        
        if result.success:
            print(f"   Found {len(result.data)} AI/ML documents above 20% similarity")
            
            # Group by tags
            tag_counts = {}
            for hit in result.data:
                for tag in hit["metadata"]["tags"]:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            print("   Most common tags:")
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"     • {tag}: {count} documents")
        
        # Performance statistics
        print("\n7. Search performance statistics...")
        stats = db.get_collection_stats("documents")
        
        print(f"   📊 Total documents: {stats.total_vectors}")
        print(f"   📊 Index type: {stats.index_type}")
        print(f"   📊 Average search latency: {stats.avg_search_latency_ms:.2f}ms")
        print(f"   📊 Storage size: {stats.size_bytes:,} bytes")
        
        # Recommendation system simulation
        print("\n8. Document recommendation system...")
        
        # Pick a document as "user interest"
        base_doc = SAMPLE_DOCUMENTS[1]  # Machine Learning Fundamentals
        print(f"   Based on interest in: '{base_doc['title']}'")
        
        # Use its embedding for recommendations
        base_embedding = simulate_sentence_embeddings(
            f"{base_doc['title']} {base_doc['content']}", 384
        )
        
        recommendation_query = SearchQuery(
            vector=base_embedding,
            k=4,  # Exclude the original document
            include_metadata=True,
            threshold=0.1
        )
        
        rec_result = await db.search_vectors("documents", recommendation_query)
        
        if rec_result.success:
            print("   Recommended documents:")
            for i, rec in enumerate(rec_result.data[1:], 1):  # Skip first (self)
                title = rec["metadata"]["title"]
                similarity = rec["score"] * 100
                category = rec["metadata"]["category"]
                print(f"   {i}. {title} ({category}) - {similarity:.1f}% similar")
        
        print("\n" + "=" * 50)
        print("🎯 Semantic Search System Complete!")
        print("\nFeatures demonstrated:")
        print("  • Document embedding and indexing")
        print("  • Natural language query processing")
        print("  • Category-based filtering")
        print("  • Multi-criteria search")
        print("  • Recommendation system")
        print("  • Performance monitoring")
        
    finally:
        await db.close()
        print(f"\n📁 Search index saved at {db_path}")


async def main():
    """Run the semantic search example."""
    try:
        await build_semantic_search()
        
        print("\n🌟 Try modifying the example to:")
        print("  • Add more documents")
        print("  • Use real embedding models")
        print("  • Implement user feedback")
        print("  • Add temporal filtering")
        print("  • Build a web interface")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
