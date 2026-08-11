# Observable LLM pipelines and FAISS discovery

ToucanDB provides the retrieval and memory part of an LLM pipeline without
requiring a pipeline framework or a database server. Models, parsers, access
control, and product decisions remain application-owned.

## How the seven stages map to ToucanDB

| Pipeline stage | ToucanDB responsibility | Application responsibility |
|---|---|---|
| Ingestion | Stable document IDs, idempotent sync, namespace pruning | PDF, email, web, OCR, and business-system loaders |
| Preprocessing | Deterministic overlapping chunks and injected embeddings | Model choice, language handling, sensitive-data policy |
| Context | Namespace-safe retrieval, metadata filters, bounded evidence blocks | Authorization, query policy, product-specific reranking |
| Inference | Sync/async generator contract | Hosted or local LLM, credentials, rate limits |
| Postprocessing | Optional traced postprocessor | Schema validation, domain rules, human-review gates |
| Integration | Python API and SimpliXio adapter | UI, workflows, source-record synchronization |
| Evaluation | Hit rate, recall@k, MRR, latency, privacy-safe traces | Reviewed relevance labels, answer-quality rubric, alerts |

This separation avoids making LangChain, LlamaIndex, FastAPI, Docker, or a
hosted model mandatory. Those tools can still be used around ToucanDB when a
deployment benefits from them.

## Run an observable pipeline

`RAGPipeline` records stage timings, retrieved IDs and scores, model identity,
and bounded output sizes. It intentionally does not put the raw question,
prompt, retrieved text, or answer in the trace.

```python
from toucandb.integrations import RAGPipeline


async def generate(prompt: str) -> str:
    return await llm.generate(prompt)


def validate_output(answer: str) -> str:
    if not answer.strip():
        raise ValueError("empty model answer")
    return answer.strip()


pipeline = RAGPipeline(rag)
result = await pipeline.run(
    "How does recovery work?",
    generate,
    k=6,
    postprocessor=validate_output,
)

print(result.answer.answer)
print(result.trace.total_ms)
print(result.trace.stages)
```

An optional sync or async observer can forward completed success and failure
traces to the application's metrics system. Observer failures are logged and
cannot turn a successful product request into a failed one. Retrieved IDs can
still be sensitive provenance, so traces need the same retention and access
policy as application logs.

## Evaluate retrieval separately from generation

An LLM judge cannot tell whether the retriever consistently found the reviewed
source of truth. Maintain a small, versioned set of real queries and relevant
document IDs, then measure retrieval before evaluating writing quality.

```python
from toucandb.integrations import RetrievalEvaluationCase

report = await pipeline.evaluate_retrieval(
    [
        RetrievalEvaluationCase(
            query="How is a stale index recovered?",
            relevant_document_ids=frozenset({"architecture"}),
        ),
        RetrievalEvaluationCase(
            query="When is a backend required?",
            relevant_document_ids=frozenset({"apple-integration"}),
        ),
    ],
    k=5,
)

print(report.hit_rate_at_k)
print(report.mean_recall_at_k)
print(report.mean_reciprocal_rank)
print(report.p95_latency_ms)
```

Use exact `flat` search as a recall baseline when tuning HNSW or IVF. Evaluate
authorization filters independently: semantic similarity is not permission to
see a record. Evaluation reports intentionally retain their reviewed queries;
build them from a controlled test set rather than copying live user traffic.

## FAISS search versus clustering

These operations solve different problems:

- **Nearest-neighbor search** starts with a query vector and returns the closest
  stored vectors. ToucanDB uses this for semantic search and RAG.
- **Clustering** has no query. It groups a corpus around learned centroids so an
  application can explore themes, detect duplicate neighborhoods, or choose a
  stratified sample for human review.
- **IVF indexing** uses clustering internally to partition search candidates.
  Its centroids optimize retrieval speed; they should not automatically be
  presented as meaningful product topics.

ToucanDB exposes non-persistent semantic discovery separately:

```python
clusters = await pipeline.discover_clusters(
    12,
    metadata_filter={"visibility": "public"},
    representative_count=3,
    max_vectors=50_000,
    max_memory_bytes=512 * 1024 * 1024,
)

for cluster in clusters.clusters:
    print(cluster.member_count, cluster.document_ids)
    print([hit.source for hit in cluster.representative_hits])
```

Cosine collections use spherical FAISS k-means. Other metrics use standard
FAISS k-means. Snapshot loading, training, and assignment run off the event
loop. Vector-count and working-memory budgets fail before an unbounded
vector-matrix allocation. They do not estimate every Python metadata or text
object, so `max_vectors` remains a second independent bound. Clusters are not
persisted because they are model-derived views: changing the embedding model,
corpus, filter, or seed can change the groups. Source records remain the source
of truth.

## SimpliXio deployment decision

For private notes on one iPhone or Mac, SimpliXio should continue using Apple
Natural Language, SQLite WAL, Accelerate, and Swift actors. Exact vector search
is fast and operationally simple at the current on-device scale. Packaging
Python and FAISS into the app would add size and signing complexity without a
corresponding product benefit.

An optional backend becomes useful for a much larger shared corpus, hosted LLM
credentials, centralized evaluation, collaboration, or cross-account access
control. In that configuration:

1. Devices synchronize source records and provenance, not embedding vectors.
2. The server owns one ToucanDB process and its FAISS accelerator.
3. Each device keeps its private, low-latency native index.
4. Retrieval results can be merged only after authorization and sensitivity
   policy, with source attribution preserved.

This creates a local-first evidence pipeline rather than another opaque
chatbot: deterministic product logic remains authoritative, semantic retrieval
supplies candidates, and every generated answer can carry inspectable evidence
and measurable retrieval quality.

## Primary references

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [FAISS project and documentation](https://github.com/facebookresearch/faiss)
- [FAISS k-means implementation notes](https://github.com/facebookresearch/faiss/wiki/Implementation-notes)
- [Apple: Finding similarities between pieces of text](https://developer.apple.com/documentation/naturallanguage/finding-similarities-between-pieces-of-text)
