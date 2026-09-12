# 🦜 ToucanDB

**Store, index, and search high-dimensional vector embeddings. Built for RAG systems, semantic search, and LLM applications.**

<p align="center">
  <a href="https://github.com/ToucanDB/ToucanDB/blob/main/assets/promo/toucandb-promo.mp4">
    <img src="https://raw.githubusercontent.com/ToucanDB/ToucanDB/main/assets/promo/toucandb-promo-poster.png" width="100%" alt="Watch the 18-second ToucanDB product overview">
  </a>
</p>

<p align="center"><strong>▶ Watch the 18-second ToucanDB overview — with sound</strong></p>

An embedded, local-first vector database for semantic search, RAG, and
application memory. ToucanDB combines atomic SQLite persistence with FAISS
search and does not require a separate database server.

[![PyPI](https://img.shields.io/pypi/v/toucandb.svg)](https://pypi.org/project/toucandb/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/ToucanDB/ToucanDB/actions/workflows/ci.yml/badge.svg)](https://github.com/ToucanDB/ToucanDB/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.com/ToucanDB/ToucanDB/blob/main/LICENSE.md)

## 📋 Main features in version 2

- Cosine, dot-product, and Euclidean similarity.
- Exact flat search and approximate HNSW or IVF indices.
- SQLite WAL storage with atomic batch writes and safe arbitrary vector IDs.
- Generation-checked FAISS snapshots with deterministic rebuild after a crash
  or a cross-architecture restore.
- Bounded, database-wide LRU cache budgets and automatic tombstone compaction.
- Optional authenticated encryption with a per-database salt and memory-hard
  key derivation.
- Async APIs that move storage, index, and local-model work off the event loop.
- Stable upserts that do not re-embed unchanged documents.
- A dependency-free RAG pipeline with chunking, namespace isolation, pruning,
  source attribution, bounded context, and an injected generator.
- Privacy-safe pipeline traces, reviewed retrieval evaluation, and optional
  bounded FAISS topic discovery.
- Lazy Sentence Transformers and OpenAI embedding adapters, plus a generic
  callable adapter.
- A tested SimpliXio semantic-signal integration.

ToucanDB is an embedded component, not a distributed service. One process owns
a database directory at a time; a second owner fails fast instead of operating
on a stale in-memory index.

## Installation

ToucanDB 2 requires Python 3.10 or newer.

```bash
pip install toucandb
```

Install only the integration you use:

```bash
pip install 'toucandb[embeddings]'  # local Sentence Transformers
pip install 'toucandb[openai]'      # OpenAI embeddings
pip install 'toucandb[all]'         # both adapters
pip install 'toucandb[dev]'         # contributors and release checks
```

The RAG pipeline itself is in the core package and adds no model or framework
dependency. The old `ml` extra remains as an alias for `embeddings` during the
2.x transition. ToucanDB does not publish a `gpu` extra because the upstream
FAISS GPU wheel has been discontinued; custom FAISS GPU builds are an advanced
deployment choice.

## RAG quick start

```python
import asyncio

from toucandb import SentenceTransformerEmbeddingProvider
from toucandb.integrations import RAGDocument, RAGStore


async def main() -> None:
    rag = await RAGStore.create(
        "./knowledge.tdb",
        SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2"),
        encryption_key="load-this-from-a-secret-store",
        namespace="product-docs",
    )

    await rag.sync_documents(
        [
            RAGDocument(
                id="architecture",
                text="ToucanDB stores records in SQLite and rebuilds FAISS safely.",
                source="architecture.md",
                metadata={"project": "ToucanDB"},
            ),
            RAGDocument(
                id="deployment",
                text="ToucanDB is embedded and needs one owning process.",
                source="deployment.md",
                metadata={"project": "ToucanDB"},
            ),
        ],
        prune=True,
    )

    hits = await rag.retrieve("Does ToucanDB need a database server?", k=3)
    for hit in hits:
        print(hit.score, hit.source, hit.text)

    # `answer()` accepts any async object with generate(prompt), or a callable.
    async def generator(prompt: str) -> str:
        return await your_llm(prompt)

    answer = await rag.answer(
        "Does ToucanDB need a database server?",
        generator,
    )
    print(answer.answer)
    await rag.close()


asyncio.run(main())
```

See [the RAG guide](https://github.com/ToucanDB/ToucanDB/blob/main/docs/rag.md) and
[`examples/rag_pipeline.py`](https://github.com/ToucanDB/ToucanDB/blob/main/examples/rag_pipeline.py) for a runnable example.
Production tracing, recall/MRR evaluation, postprocessing, and semantic
clustering are covered in [the LLM pipeline guide](https://github.com/ToucanDB/ToucanDB/blob/main/docs/llm-pipelines.md).

## Raw-vector quick start

```python
import asyncio

from toucandb import SearchQuery, ToucanDB, create_schema


async def main() -> None:
    async with await ToucanDB.create("./vectors.tdb") as db:
        await db.create_collection(
            create_schema("items", dimensions=3, index_type="hnsw")
        )
        result = await db.upsert_vectors(
            "items",
            [
                {
                    "id": "toucan",
                    "vector": [0.9, 0.1, 0.2],
                    "metadata": {"kind": "bird"},
                },
                {
                    "id": "macaw",
                    "vector": [0.8, 0.2, 0.1],
                    "metadata": {"kind": "bird"},
                },
            ],
        )
        if not result.success:
            raise RuntimeError(result.error_message)

        matches = await db.search_vectors(
            "items",
            SearchQuery(
                vector=[1.0, 0.0, 0.0],
                k=2,
                metadata_filter={"kind": "bird"},
            ),
        )
        print(matches.data)


asyncio.run(main())
```

## Architecture and resource use

```text
application
   │
   ├── vectors supplied directly, or lazy embedding provider
   │
   ▼
ToucanDB collection
   ├── SQLite WAL: durable source of truth and atomic batches
   ├── FAISS: in-memory search accelerator
   ├── snapshot manifest: generation + schema verification
   └── bounded LRU: vectors/metadata loaded by results
```

The design deliberately avoids a background server, a connection pool, an
unbounded cache, and eager model loading. SQLite records are authoritative;
FAISS can always be rebuilt. A clean close writes an index snapshot. A crash
after a committed data write produces a generation mismatch, so the next open
rebuilds instead of trusting stale search state.

| Index | Best fit | Trade-off |
|---|---|---|
| `flat` | Small collections or exact evaluation | Exact recall; linear search |
| `hnsw` | Default interactive retrieval | Fast approximate search; extra memory |
| `ivf` | Larger, batch-oriented corpora | Trained approximate index; tune `nprobe` |

Metadata filters use adaptive candidate expansion. Selective filters may
require more index work; benchmark with the metadata distribution and recall
requirements of the real corpus.

See [architecture](https://github.com/ToucanDB/ToucanDB/blob/main/docs/architecture.md),
[performance guidance](https://github.com/ToucanDB/ToucanDB/blob/main/docs/performance.md), and the
[2.0 migration guide](https://github.com/ToucanDB/ToucanDB/blob/main/docs/migration-2.0.md). FAISS topic discovery is a
separate, non-persistent analysis operation; IVF's internal centroids are search
partitions and should not be treated as product labels.

## Does it need a backend?

No backend is required when one Python application owns local data. This is the
best fit for desktop tools, local agents, evaluation pipelines, private RAG,
and an application service that embeds ToucanDB in its own process.

A service boundary does make sense when multiple processes or devices need the
same live index, when API credentials must not ship in a client, or when access
control and synchronization are server responsibilities. In that scenario,
run one ToucanDB owner behind the application's API; do not let several workers
write the same directory.

For iOS and macOS, the Python wheel is not embedded into the app. SimpliXio uses
a native Swift runtime built from Apple Natural Language, SQLite WAL,
Accelerate, and actor isolation. It synchronizes source records rather than
model-specific vectors and needs no backend for an on-device-only experience.
See [Apple integration](https://github.com/ToucanDB/ToucanDB/blob/main/docs/apple-integration.md).

## Security boundary

Pass an encryption key only from a keychain, secret manager, or environment
owned by the host application—never hard-code it. Encryption covers vector and
metadata payloads. Collection names, vector IDs, hashed record keys, SQLite
structure, and index snapshots are not encrypted. If whole-index
confidentiality matters, rely on full-disk/platform data protection as well. A
lost encryption key cannot be recovered.

Retrieved RAG text is untrusted input. `RAGStore.answer()` tells the generator
to treat sources as data, but the host application must still apply its normal
prompt-injection, authorization, sensitivity, and output controls.

## Performance claims

ToucanDB does not claim a universal latency, throughput, or corpus-size number.
Results depend on hardware, dimensions, index type, vector count, filters,
encryption, and recall targets. The repository includes a reproducible local
benchmark command; publish the full configuration with any reported result.

```bash
python benchmarks/benchmark.py --vectors 10000 --dimensions 384 --index hnsw
```

## SimpliXio integration

`SimplixioSignalMemory` indexes stable SimpliXio signal IDs, skips unchanged
embeddings, filters by project, and prunes deleted records with a bulk
transaction. SimpliXio keeps deterministic product ranking and sensitivity
rules as the source of truth; ToucanDB supplies semantic candidates.

See [the reviewed use case](https://github.com/ToucanDB/ToucanDB/blob/main/docs/real-project-use-cases.md) and
[`examples/simplixio_signal_memory.py`](https://github.com/ToucanDB/ToucanDB/blob/main/examples/simplixio_signal_memory.py).

## Project status and limits

ToucanDB 2 is a beta embedded engine. It is not a distributed vector database,
does not provide HTTP/authentication/multi-tenancy by itself, and does not yet
provide a transactional metadata secondary index. Equality metadata filters
are supported. Validate recall and failure behavior with representative data
before relying on it for a high-stakes production workflow.

## Author and credits

ToucanDB was created and is maintained by **Pierre-Henry Soria**:

- Website: [pierrehenry.dev](https://pierrehenry.dev)
- GitHub: [github.com/pH-7](https://github.com/pH-7)
- LinkedIn: [linkedin.com/in/ph7enry](https://www.linkedin.com/in/ph7enry/)

Additional acknowledgements and citation details are in
[CREDITS.md](https://github.com/ToucanDB/ToucanDB/blob/main/CREDITS.md) and
[CITATION.cff](https://github.com/ToucanDB/ToucanDB/blob/main/CITATION.cff).

## Why “ToucanDB”?

Toucans reflect the project’s aim: precise retrieval, adaptability across data
sources, and a vivid local-first identity. Pierre-Henry’s family history in the
Amazon and lifelong love of birds inspired the name.

![Pierre-Henry with toucans](https://raw.githubusercontent.com/ToucanDB/ToucanDB/main/assets/why-its-called-toucandb.jpeg "Why ToucanDB is named after toucans")

## Contributing and license

See [CONTRIBUTING.md](https://github.com/ToucanDB/ToucanDB/blob/main/CONTRIBUTING.md) for the development workflow. ToucanDB is
released under the [MIT License](https://github.com/ToucanDB/ToucanDB/blob/main/LICENSE.md).

---

**Built with ❤️ for the AI community**
