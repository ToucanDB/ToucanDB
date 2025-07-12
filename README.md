# 🦜 ToucanDB - Micro ML-First Vectorial DB Engine

**A Secure, Efficient ML-First Vector Database Engine**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/pH-7/toucandb/workflows/Tests/badge.svg)](https://github.com/pH-7/toucandb/actions)

ToucanDB is a modern, security-first vector database designed specifically for ML workloads. Named after the vibrant toucan bird known for its precision and adaptability, ToucanDB brings the same qualities to vector similarity search and storage.

## 🌟 Key Features

### 🔒 Security-First Design
- **End-to-end encryption** with AES-256-GCM
- **Zero-trust architecture** with granular access controls
- **Audit logging** for compliance and monitoring
- **Memory-safe operations** to prevent data leaks

### ⚡ High Performance
- **SIMD-optimized** vector operations
- **Adaptive indexing** with HNSW and IVF algorithms
- **Intelligent caching** with LRU and frequency-based eviction
- **Async I/O** for concurrent operations

### 🧠 ML-Native Features
- **Multi-model support** for various embedding types
- **Automatic quantization** (FP16, INT8) for memory efficiency
- **Vector clustering** and dimensionality reduction
- **Real-time inference** integration

### 🎯 Developer Experience
- **Simple Python API** with async support
- **Type-safe** operations with Pydantic schemas
- **Comprehensive testing** and documentation
- **Docker support** for easy deployment

## 🚀 Quick Start

```python
import asyncio
from toucandb import ToucanDB, VectorSchema

async def main():
    # Initialize database
    db = await ToucanDB.create("./my_vectors.tdb", encryption_key="your-secret-key")
    
    # Define schema
    schema = VectorSchema(
        name="embeddings",
        dimensions=384,
        metric="cosine",
        index_type="hnsw"
    )
    
    # Create collection
    collection = await db.create_collection(schema)
    
    # Insert vectors
    vectors = [
        {"id": "doc1", "vector": [0.1, 0.2, ...], "metadata": {"title": "AI Paper"}},
        {"id": "doc2", "vector": [0.3, 0.4, ...], "metadata": {"title": "ML Tutorial"}},
    ]
    await collection.insert_many(vectors)
    
    # Search similar vectors
    query_vector = [0.15, 0.25, ...]
    results = await collection.search(query_vector, k=5, threshold=0.8)
    
    for result in results:
        print(f"ID: {result.id}, Score: {result.score}, Metadata: {result.metadata}")

# Run the example
asyncio.run(main())
```

## 📦 Installation

```bash
# Basic installation
pip install toucandb

# With GPU support
pip install toucandb[gpu]

# Development installation
pip install toucandb[dev]
```

## 🏗️ Architecture

ToucanDB is built on a modular architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client API    │    │  Security Layer │    │  Query Engine   │
│                 │    │                 │    │                 │
│ • Python SDK    │    │ • Encryption    │    │ • Vector Search │
│ • Async Support │    │ • Access Control│    │ • Filtering     │
│ • Type Safety   │    │ • Audit Logs    │    │ • Aggregation   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Index Engine   │    │  Storage Engine │    │  Memory Manager │
│                 │    │                 │    │                 │
│ • HNSW/IVF      │    │ • Compressed    │    │ • Smart Caching │
│ • Auto-tuning   │    │ • Transactional │    │ • Memory Pools  │
│ • Multi-metric  │    │ • Backup/Restore│    │ • Garbage Collection │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

ToucanDB supports extensive configuration options:

```python
config = {
    "storage": {
        "compression": "lz4",
        "encryption": "aes-256-gcm",
        "backup_interval": 3600
    },
    "indexing": {
        "algorithm": "hnsw",
        "ef_construction": 200,
        "m": 16,
        "auto_optimize": True
    },
    "memory": {
        "cache_size": "1GB",
        "memory_map": True,
        "preload": False
    }
}

db = await ToucanDB.create("./db", config=config)
```

## 🧪 Use Cases

- **Semantic Search**: Document and image similarity search
- **Recommendation Systems**: User and item embeddings
- **RAG Applications**: Knowledge base for LLMs
- **Anomaly Detection**: Outlier identification in high-dimensional data
- **Clustering & Classification**: ML feature stores

## 🛡️ Security Features

### Encryption
- All data encrypted at rest and in transit
- Key rotation and management
- Hardware security module (HSM) support

### Access Control
- Role-based permissions
- API key authentication
- Network-level security

### Compliance
- GDPR-compliant data handling
- SOC 2 Type II controls
- Comprehensive audit trails

## 📈 Performance

ToucanDB is designed for high performance:

- **Throughput**: 100K+ vectors/second ingestion
- **Latency**: Sub-millisecond search response times
- **Scale**: Billions of vectors with horizontal scaling
- **Memory**: Optimized memory usage with smart compression

## 👨‍💻 Who Built This Vector Database Engine?

**Pierre-Henry Soria** — a **super passionate engineer** who loves building cutting-edge AI infrastructure and automating intelligent systems efficiently!
Enthusiast of Machine Learning, Vector Databases, AI, and—of course—writing performant code!
Find me at [pH7.me](https://ph7.me)

Enjoying this project? **[Buy me a coffee](https://ko-fi.com/phenry)** (spoiler: I love almond extra-hot flat white coffees while coding ML algorithms).

[![Pierre-Henry Soria](https://s.gravatar.com/avatar/a210fe61253c43c869d71eaed0e90149?s=200)](https://ph7.me "Pierre-Henry Soria's personal website")

[![@phenrysay][x-icon]](https://x.com/phenrysay "Follow Me on X") [![YouTube Tech Videos][youtube-icon]](https://www.youtube.com/@pH7Programming "My YouTube Tech Channel") [![pH-7][github-icon]](https://github.com/pH-7 "Follow Me on GitHub")

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

ToucanDB is released under the MIT License. See [LICENSE](LICENSE) for details.

## 🎯 Roadmap

- [ ] Distributed clustering support
- [ ] Real-time streaming updates
- [ ] Multi-modal search (text + image)
- [ ] Integration with popular ML frameworks
- [ ] Cloud-native deployment options
- [ ] GraphQL API support

---

**Built with ❤️ for the AI community**

<!-- GitHub's Markdown reference links -->
[x-icon]: https://img.shields.io/badge/x-000000?style=for-the-badge&logo=x
[github-icon]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
[youtube-icon]: https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white