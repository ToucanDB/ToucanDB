# RAG integration

`RAGStore` is ToucanDB's framework-neutral retrieval-augmented generation
layer. Chunking, indexing, retrieval, context construction, and synchronization
are included in the core package. Embedding and generation are injected.

## Choose an embedding provider

Local Sentence Transformers keeps source text on the machine and loads the
model lazily:

```bash
pip install 'toucandb[embeddings]'
```

```python
from toucandb import SentenceTransformerEmbeddingProvider

provider = SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2")
```

OpenAI embeddings are also lazy and require only their own extra:

```bash
pip install 'toucandb[openai]'
```

```python
from toucandb import OpenAIEmbeddingProvider

provider = OpenAIEmbeddingProvider("text-embedding-3-small")
```

The OpenAI client reads its normal environment configuration when no client is
injected. Do not put an API key in source code or ship a privileged key in a
client app.

Any existing model SDK can be adapted without ToucanDB depending on it:

```python
from toucandb import CallableEmbeddingProvider

provider = CallableEmbeddingProvider(
    my_async_embed_function,
    dimensions=768,
    model_id="my-model:v3",
)
```

`model_id` matters. ToucanDB stores it with documents, skips model calls when
text/metadata/model identity are unchanged, and re-embeds if the model identity
changes.

## Index documents

```python
from toucandb.integrations import RAGDocument, RAGStore, TextChunker

rag = await RAGStore.create(
    "./docs.tdb",
    provider,
    namespace="public-docs",
    chunker=TextChunker(chunk_size=1200, chunk_overlap=160),
)

result = await rag.sync_documents(
    [
        RAGDocument(
            id="install",
            text="...",
            source="install.md",
            metadata={"product": "ToucanDB", "visibility": "public"},
        )
    ],
    prune=True,
)
assert result.success, result.error_message
print(result.data)
```

Document IDs must be stable. Chunk IDs are derived from the namespace,
document ID, and chunk position. `prune=True` means the supplied documents are
a complete snapshot for this namespace; chunks no longer present are removed
in one transaction. Other namespaces sharing the collection are untouched.

`sync_paths()` provides a lightweight UTF-8 text loader. For PDF, Office,
web-page, OCR, or structured-data parsing, use a dedicated parser and pass its
clean text and provenance as `RAGDocument`. Keeping parsers outside the core
avoids installing resources a project does not use.

## Retrieve and filter

```python
hits = await rag.retrieve(
    "How is the index recovered after a crash?",
    k=6,
    metadata_filter={"visibility": "public"},
    min_score=0.25,
)

for hit in hits:
    print(hit.source, hit.score, hit.text)
```

The namespace filter is always added by `RAGStore`; callers cannot accidentally
retrieve another namespace. User filters are equality filters. Scores depend on
the embedding model and metric—calibrate `min_score` on reviewed examples
instead of copying a universal threshold.

## Generate an answer

```python
class Generator:
    async def generate(self, prompt: str) -> str:
        return await llm_client.generate(prompt)


answer = await rag.answer(
    "How is the index recovered after a crash?",
    Generator(),
    k=6,
    max_context_characters=12_000,
)

print(answer.answer)
print(answer.hits)  # exact evidence supplied to the generator
```

A sync/async callable is accepted as well. Context blocks are numbered, source
attributed, and bounded. The default prompt requires evidence citations and
treats retrieved source text as untrusted data.

## Safety and evaluation

- Enforce authorization and sensitivity filters before retrieval; semantic
  similarity is not an access-control mechanism.
- Treat source content as potentially adversarial prompt input.
- Log source IDs and model identity so an answer can be reproduced.
- Evaluate retrieval recall separately from answer quality.
- Keep a deterministic fallback when RAG is an enhancement to critical product
  behavior.
- Use a backend when a hosted model key must not be present on the device.
