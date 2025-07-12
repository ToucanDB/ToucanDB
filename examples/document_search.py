#!/usr/bin/env python3
"""
Real-world ToucanDB Usage Example

This example shows how to use ToucanDB for a practical application:
building a document search system with actual text processing.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from toucandb import ToucanDB, create_schema, SearchQuery


def simple_text_vectorizer(text: str, dimensions: int = 256) -> List[float]:
    """
    A simple text vectorizer using character frequency analysis.
    In production, use proper embedding models like sentence-transformers.
    """
    import math
    
    # Normalize text
    text = text.lower().strip()
    
    # Character frequency analysis
    char_freq = {}
    for char in text:
        if char.isalnum():
            char_freq[char] = char_freq.get(char, 0) + 1
    
    # Create base vector from character frequencies
    vector = [0.0] * dimensions
    
    # Fill vector with character frequency patterns
    for i, char in enumerate('abcdefghijklmnopqrstuvwxyz0123456789'):
        if i < dimensions:
            freq = char_freq.get(char, 0)
            vector[i] = freq / max(len(text), 1)
    
    # Add word-based features
    words = text.split()
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
    
    if dimensions > 40:
        vector[36] = word_count / 100.0  # Normalized word count
        vector[37] = avg_word_length / 10.0  # Normalized avg word length
        vector[38] = len(text) / 1000.0  # Normalized text length
        vector[39] = len(set(words)) / max(word_count, 1)  # Vocabulary diversity
    
    # Add semantic features based on keywords
    semantic_keywords = {
        'technology': [50, 60],
        'programming': [51, 61], 
        'database': [52, 62],
        'artificial intelligence': [53, 63],
        'machine learning': [54, 64],
        'data science': [55, 65],
        'web development': [56, 66],
        'security': [57, 67],
        'cloud': [58, 68],
        'software': [59, 69]
    }
    
    for keyword, indices in semantic_keywords.items():
        if keyword in text and all(i < dimensions for i in indices):
            for idx in indices:
                vector[idx] = 1.0
    
    # Normalize vector
    magnitude = math.sqrt(sum(x*x for x in vector))
    if magnitude > 0:
        vector = [x / magnitude for x in vector]
    
    return vector


async def document_search_system():
    """Demonstrate a practical document search system."""
    print("📚 ToucanDB Document Search System")
    print("=" * 50)
    
    # Sample documents (could be loaded from files, databases, etc.)
    documents = [
        {
            "id": "doc_001",
            "title": "Getting Started with Python Programming",
            "content": "Python is a versatile programming language that is widely used for web development, data analysis, artificial intelligence, and automation. This guide covers the basics of Python syntax, data types, and control structures.",
            "author": "Alice Johnson",
            "category": "Programming",
            "publish_date": "2024-01-15",
            "tags": ["python", "programming", "beginner", "tutorial"]
        },
        {
            "id": "doc_002", 
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. Key concepts include supervised learning, unsupervised learning, and neural networks.",
            "author": "Bob Smith",
            "category": "AI/ML",
            "publish_date": "2024-01-20",
            "tags": ["machine learning", "AI", "data science", "algorithms"]
        },
        {
            "id": "doc_003",
            "title": "Database Design Best Practices",
            "content": "Effective database design is crucial for building scalable applications. This article covers normalization, indexing strategies, query optimization, and choosing between SQL and NoSQL databases for different use cases.",
            "author": "Carol Chen",
            "category": "Database",
            "publish_date": "2024-02-01",
            "tags": ["database", "SQL", "NoSQL", "design patterns"]
        },
        {
            "id": "doc_004",
            "title": "Web Security Fundamentals",
            "content": "Web application security is essential for protecting user data and preventing cyber attacks. Learn about common vulnerabilities like XSS, CSRF, SQL injection, and how to implement secure authentication and authorization.",
            "author": "David Wilson",
            "category": "Security",
            "publish_date": "2024-02-10",
            "tags": ["security", "web development", "cybersecurity", "authentication"]
        },
        {
            "id": "doc_005",
            "title": "Cloud Computing Architecture",
            "content": "Cloud computing provides scalable and cost-effective solutions for modern applications. Explore different cloud service models (IaaS, PaaS, SaaS), deployment strategies, and best practices for building cloud-native applications.",
            "author": "Eve Taylor",
            "category": "Cloud",
            "publish_date": "2024-02-15",
            "tags": ["cloud computing", "AWS", "architecture", "scalability"]
        },
        {
            "id": "doc_006",
            "title": "Data Visualization with Python",
            "content": "Data visualization is key to understanding complex datasets and communicating insights effectively. This tutorial covers popular Python libraries like Matplotlib, Seaborn, and Plotly for creating interactive charts and graphs.",
            "author": "Frank Brown",
            "category": "Data Science",
            "publish_date": "2024-02-20",
            "tags": ["data visualization", "python", "matplotlib", "analytics"]
        }
    ]
    
    # Create database
    db_path = Path("./document_search_db")
    db = await ToucanDB.create(db_path, encryption_key="document-search-key")
    
    try:
        # Create collection
        print("\n1. Setting up document collection...")
        schema = create_schema(
            name="documents",
            dimensions=256,
            metric="cosine",
            index_type="hnsw"
        )
        
        collection = await db.create_collection(schema)
        print(f"   ✓ Created collection with {schema.dimensions}D vectors")
        
        # Process and index documents
        print("\n2. Processing and indexing documents...")
        vectors = []
        
        for doc in documents:
            # Combine title and content for vectorization
            full_text = f"{doc['title']} {doc['content']}"
            
            # Generate vector representation
            vector = simple_text_vectorizer(full_text, 256)
            
            vectors.append({
                "id": doc["id"],
                "vector": vector,
                "metadata": {
                    "title": doc["title"],
                    "author": doc["author"],
                    "category": doc["category"],
                    "publish_date": doc["publish_date"],
                    "tags": doc["tags"],
                    "content_preview": doc["content"][:200] + "...",
                    "word_count": len(doc["content"].split()),
                    "content_length": len(doc["content"])
                }
            })
        
        # Insert documents
        result = await db.insert_vectors("documents", vectors)
        if result.success:
            print(f"   ✓ Indexed {len(result.data)} documents")
        else:
            print(f"   ✗ Indexing failed: {result.error_message}")
            return
        
        # Interactive search system
        print("\n3. Interactive document search...")
        
        search_queries = [
            "Python programming tutorial for beginners",
            "Machine learning algorithms and data science",
            "Database optimization and performance",
            "Web application security best practices",
            "Cloud infrastructure and scalability",
            "Data visualization and analytics"
        ]
        
        for i, query_text in enumerate(search_queries, 1):
            print(f"\n   Query {i}: '{query_text}'")
            print("   " + "-" * 50)
            
            # Vectorize the search query
            query_vector = simple_text_vectorizer(query_text, 256)
            
            # Search for relevant documents
            search_query = SearchQuery(
                vector=query_vector,
                k=3,
                include_metadata=True,
                threshold=0.1
            )
            
            search_result = await db.search_vectors("documents", search_query)
            
            if search_result.success and search_result.data:
                for j, hit in enumerate(search_result.data, 1):
                    score = hit["score"]
                    metadata = hit["metadata"]
                    
                    # Display result
                    print(f"   {j}. 📄 {metadata['title']}")
                    print(f"      👤 Author: {metadata['author']}")
                    print(f"      🏷️  Category: {metadata['category']}")
                    print(f"      📅 Published: {metadata['publish_date']}")
                    print(f"      🎯 Relevance: {score:.3f}")
                    print(f"      📝 Preview: {metadata['content_preview']}")
                    print(f"      🏷️  Tags: {', '.join(metadata['tags'])}")
                    print()
            else:
                print("   No relevant documents found.")
        
        # Category-based filtering
        print("\n4. Category-based search...")
        query_text = "programming and software development"
        query_vector = simple_text_vectorizer(query_text, 256)
        
        categories = ["Programming", "AI/ML", "Database", "Security"]
        
        for category in categories:
            print(f"\n   🔍 Searching in '{category}' category:")
            
            filtered_query = SearchQuery(
                vector=query_vector,
                k=2,
                include_metadata=True,
                metadata_filter={"category": category}
            )
            
            result = await db.search_vectors("documents", filtered_query)
            
            if result.success and result.data:
                best_match = result.data[0]
                print(f"   📄 {best_match['metadata']['title']}")
                print(f"   🎯 Relevance: {best_match['score']:.3f}")
            else:
                print("   No matches found in this category")
        
        # Author-based search
        print("\n5. Author-based recommendations...")
        
        # Find documents by specific author
        author_query = SearchQuery(
            vector=[0.0] * 256,  # Dummy vector since we're filtering by metadata
            k=10,
            include_metadata=True,
            metadata_filter={"author": "Alice Johnson"}
        )
        
        author_result = await db.search_vectors("documents", author_query)
        
        if author_result.success and author_result.data:
            print(f"   📚 Documents by Alice Johnson:")
            for doc in author_result.data:
                print(f"   • {doc['metadata']['title']}")
        
        # Statistics and analytics
        print("\n6. Search system analytics...")
        
        stats = db.get_collection_stats("documents")
        db_info = db.get_database_info()
        
        print(f"   📊 Total documents: {stats.total_vectors}")
        print(f"   📊 Storage size: {stats.size_bytes:,} bytes")
        print(f"   📊 Average search latency: {stats.avg_search_latency_ms:.2f}ms")
        print(f"   📊 Database version: {db_info['version']}")
        print(f"   📊 Collections: {db_info['total_collections']}")
        
        # Tag analysis
        print("\n7. Content analysis...")
        
        # Get all documents and analyze tags
        all_docs_query = SearchQuery(
            vector=[0.0] * 256,
            k=100,
            include_metadata=True
        )
        
        all_docs_result = await db.search_vectors("documents", all_docs_query)
        
        if all_docs_result.success:
            # Collect tag statistics
            tag_counts = {}
            category_counts = {}
            total_words = 0
            
            for doc in all_docs_result.data:
                metadata = doc["metadata"]
                
                # Count tags
                for tag in metadata["tags"]:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                # Count categories
                category = metadata["category"]
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Sum word counts
                total_words += metadata["word_count"]
            
            print(f"   📈 Most common tags:")
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      • {tag}: {count} documents")
            
            print(f"   📈 Category distribution:")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"      • {category}: {count} documents")
            
            print(f"   📈 Total words indexed: {total_words:,}")
            print(f"   📈 Average words per document: {total_words / len(all_docs_result.data):.1f}")
        
        print("\n" + "=" * 50)
        print("🎯 Document Search System Complete!")
        print("\nSystem capabilities:")
        print("  • Full-text semantic search")
        print("  • Category and metadata filtering")
        print("  • Author-based recommendations")
        print("  • Real-time analytics and insights")
        print("  • Scalable vector indexing")
        print("  • Encrypted data storage")
        
    finally:
        await db.close()
        print(f"\n💾 Search index persisted at {db_path}")


async def main():
    """Run the document search system example."""
    try:
        await document_search_system()
        
        print("\n🌟 Next Steps:")
        print("  • Integrate real embedding models (sentence-transformers)")
        print("  • Add web interface for search")
        print("  • Implement user feedback and learning")
        print("  • Scale with multiple collections")
        print("  • Add real-time document ingestion")
        
    except Exception as e:
        print(f"\n❌ System failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
