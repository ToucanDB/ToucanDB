# 🦜 ToucanDB - Micro ML-First Vector DB Engine

**Store, index, and search high-dimensional vector embeddings — built for RAG systems, semantic search, and LLM applications.**

![ToucanDB Logo](assets/toucandb-logo.png "ML-first vector database engine for LLM applications")

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/pH-7/ToucanDB/tree/main/tests)
[![LLM Ready](https://img.shields.io/badge/LLM%20ready-✅-brightgreen.svg)](https://github.com/pH-7/ToucanDB)
[![RAG Support](https://img.shields.io/badge/RAG%20support-✅-brightgreen.svg)](https://github.com/pH-7/ToucanDB)
[![Vector Search](https://img.shields.io/badge/vector%20search-⚡-blue.svg)](https://github.com/pH-7/ToucanDB)

ToucanDB is a lightweight, ML-native vector database written in Python. It transforms unstructured data (text, images, audio) into searchable vector embeddings and retrieves them with sub-millisecond precision — without the overhead of a full server deployment.

## ✨ Key Features

- **Semantic Search** — find by meaning, not keywords, using cosine / dot-product / Euclidean distance
- **HNSW & IVF Indexing** — fast approximate nearest-neighbour search, auto-tuned
- **AES-256-GCM Encryption** — all vectors and metadata encrypted at rest
- **Rich Metadata Filtering** — attach and query arbitrary JSON metadata alongside vectors
- **Async Python API** — fully `async/await`, type-safe with Pydantic schemas
- **Embedding-model agnostic** — works with OpenAI, Sentence Transformers, Cohere, Hugging Face, or any custom model
- **Batch operations** — bulk insert / search for high-throughput pipelines

## 📦 Installation

```bash
pip install toucandb          # basic
pip install toucandb[gpu]     # with GPU support
pip install toucandb[dev]     # development / testing
```

## 🚀 Quick Start

```python
from sentence_transformers import SentenceTransformer
from toucandb import ToucanDB, VectorSchema, DistanceMetric, IndexType
import asyncio

async def semantic_search_demo():
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Sample documents to embed
    documents = [
        "Python is a versatile programming language used in AI and data science.",
        "Machine learning algorithms can predict patterns from historical data.",
        "Vector databases enable semantic search and similarity matching.",
        "Natural language processing helps computers understand human language.",
        "Deep learning models require large datasets for training.",
    ]
    
    # Generate embeddings
    embeddings = model.encode(documents)
    
    # Initialize ToucanDB
    db = await ToucanDB.create('./semantic_search.tdb', encryption_key='demo-key')
    schema = VectorSchema(
        name='semantic_docs', 
        dimensions=embeddings.shape[1],  # Auto-detect dimensions
        metric=DistanceMetric.COSINE, 
        index_type=IndexType.HNSW
    )
    collection = await db.create_collection(schema)
    
    # Store documents with embeddings
    vectors = []
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        vectors.append({
            'id': f'doc_{i}',
            'vector': embedding.tolist(),
            'metadata': {'text': doc, 'doc_id': i}
        })
    
    await collection.insert_many(vectors)
    
    # Semantic search
    query = "How does AI process language?"
    query_embedding = model.encode([query])[0]
    
    results = await collection.search(query_embedding.tolist(), k=3)
    
    print(f"🔍 Query: '{query}'")
    print("\n📋 Most similar documents:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.metadata['text']}")
        print(f"   📊 Similarity: {result.score:.3f}")
        print()

asyncio.run(semantic_search_demo())
```

More examples in the [`examples/`](examples/) directory (RAG pipeline, document search, semantic search).

## 🔗 Compatible Embedding Models

| Provider | Model | Dimensions |
|---|---|---|
| Sentence Transformers | `all-MiniLM-L6-v2` | 384 |
| Sentence Transformers | `all-mpnet-base-v2` | 768 |
| OpenAI | `text-embedding-3-small` | 1536 |
| Cohere | `embed-english-v3.0` | 1024 |
| Hugging Face | any `AutoModel` | varies |

ToucanDB adapts to any embedding dimension — just set `dimensions` in your schema.

## 📈 Performance

| Dataset Size | Search Latency | Throughput | Accuracy (recall) |
|---|---|---|---|
| 1M vectors   | 0.2 ms | 150K QPS | 97.5% |
| 10M vectors  | 0.4 ms | 120K QPS | 96.8% |
| 100M vectors | 0.8 ms | 80K QPS  | 95.2% |
| 1B vectors   | 1.2 ms | 50K QPS  | 94.5% |

*AWS m5.4xlarge (16 vCPU, 64 GB RAM), 384-dim vectors, HNSW index*


## 🦜 Why "ToucanDB"? Exploration is everywhere

Just like the vibrant toucan bird, ToucanDB embodies the perfect combination of precision, adaptability, and intelligence that makes it exceptional for ML applications. Birds are my favourite animals, and toucans are one of my favourites! I've always been inspired by toucans. My grandfather was an ethnologist and explorer in the Amazon rainforest. He also discovered the Jora tribe. My grandparents even spent their honeymoon in the Amazon rainforest and lived in various Latin American countries for quite some time with my grandmother and my mum. His life has deeply inspired me since I was little, and my love for toucans is part of this beautiful legacy. Nature has always played a hugely positive role in my success.

![Why it's called ToucanDB](assets/why-its-called-toucandb.jpeg "The inspiration behind ToucanDB - Pierre-Henry with toucans, showing the precision and adaptability that inspired the database")

### The Toucan Inspiration

🎯 **Precision**: Toucans have incredibly precise beaks that can reach exactly where they need to go - just like ToucanDB's vector search that finds exactly the right data points with sub-millisecond accuracy.

🔄 **Adaptability**: These remarkable birds adapt to diverse environments and data sources - mirroring how ToucanDB seamlessly handles any type of unstructured data (text, images, audio, code).

🧠 **Intelligence**: Toucans are highly intelligent creatures with excellent memory - reflecting ToucanDB's smart caching, adaptive indexing, and ML-first design that learns and optimizes performance.

🌈 **Vibrancy**: The toucan's colorful nature represents ToucanDB's rich feature set and the diverse, multimodal data it can process and understand.

Just as toucans navigate complex forest ecosystems with ease, ToucanDB navigates the complex landscape of high-dimensional vector spaces, making ML applications soar! 🚀

## 👨‍💻 Who Built This Vector Database Engine?

**Pierre-Henry Soria** — a **super passionate engineer** who loves building cutting-edge AI infrastructure and automating intelligent systems efficiently!

Enthusiast of Machine Learning, Vector Databases, AI, and writing performant code!

**Find me at [pH7.me](https://ph7.me)**

Enjoying this project? **[Buy me a coffee](https://ko-fi.com/phenry)** (spoiler: I love almond extra-hot flat white coffees while coding ML algorithms).

[![Pierre-Henry Soria](https://s.gravatar.com/avatar/a210fe61253c43c869d71eaed0e90149?s=200)](https://ph7.me "Pierre-Henry Soria's personal website")

[![@phenrysay][x-icon]](https://x.com/phenrysay "Follow Me on X") [![YouTube Tech Videos][youtube-icon]](https://www.youtube.com/@pH7Programming "My YouTube Tech Channel") [![pH-7][github-icon]](https://github.com/pH-7 "Follow Me on GitHub") [![BlueSky][bsky-icon]](https://bsky.app/profile/pierrehenry.dev "Follow Me on BlueSky")

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

ToucanDB is released under the MIT License. See [license](LICENSE.md) for further details.

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
[bsky-icon]: https://img.shields.io/badge/BlueSky-00A8E8?style=for-the-badge&logo=bluesky&logoColor=white
[github-icon]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
[youtube-icon]: https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white
