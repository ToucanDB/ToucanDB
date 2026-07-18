# Real use cases from Pierre's existing projects

This review compared ToucanDB with the application behavior currently present
in projects under the two local `Code` folders. The useful boundary is narrow:
ToucanDB should own persistent similarity retrieval, while each application
keeps its source-of-truth records and business decisions.

## Recommendation

Use ToucanDB first as **SimpliXio's local semantic signal memory**.

SimpliXio/CortexOS is the best first integration because it is active, written
in Python, already stores structured signals, and has a specific retrieval gap:

- `cortex_core/signal_matching.py` measures recurrence and builds signal links
  with token-set Jaccard similarity.
- `ItemStore`, `KnowledgeStore`, and `InsightStore` use case-insensitive
  substring search.
- The deterministic ranking and feedback logic is already valuable and should
  remain the source of truth.

Jaccard correctly handles repeated words but misses paraphrases. For example,
"Ship the macOS release notes before TestFlight review" and "Document what
changed in the desktop build prior to beta submission" can describe the same
open loop while sharing almost no tokens.

ToucanDB now provides `SimplixioSignalMemory` for this boundary. It:

1. embeds signals locally with Sentence Transformers;
2. upserts them by stable SimpliXio signal ID;
3. encrypts vectors and metadata at rest when a key is supplied;
4. scopes retrieval by project metadata;
5. returns related records as evidence for recurrence, relationships, or
   resurfacing; and
6. prunes deleted records when synchronizing a complete JSON snapshot.

ToucanDB does **not** replace SimpliXio's JSON/SQLite records, scoring, calm
queue limits, feedback, sensitivity rules, or deterministic fallback.

## Ranked candidates

| Rank | Existing project | Concrete use | Value | Integration cost | Why now / later |
|---:|---|---|---|---|---|
| 1 | SimpliXio / CortexOSLLM | Paraphrase-aware recurrence, related-signal graph candidates, and contextual resurfacing evidence | High | Low | Active Python project; current Jaccard and substring matching expose a precise gap |
| 2 | PDF AI Streamlit | Persist encrypted PDF chunks instead of rebuilding an in-memory FAISS index on every upload | High | Low–medium | `app_pdf.py` currently calls `FAISS.from_documents`; useful if this older app is revived |
| 3 | VideoCards | Search concepts across all previously processed videos and surface related cards for review | Medium–high | Low–medium | Python app with cached results, but its present experience is organized one video at a time |
| 4 | AI Second Brain | Replace exact substring history search with "find the thought where I planned the beta" and detect duplicate goals | Medium | Medium–high | Clear user benefit, but React Native needs a Python service or a different embedded runtime |
| 5 | Learning System | Semantic RSS/YouTube deduplication and topic-diverse daily Kindle selection | Medium | Medium | Python fit; current aggregation would benefit, but ranking/dedup needs product work first |
| 6 | Muaddib | Retrieve relevant Chronicle paragraphs without loading broad memory context | High | High | Strong memory use case, but TypeScript/Python bridging and strict per-channel isolation raise the risk |
| 7 | pH7 Social Dating CMS | Profile/interest similarity candidates for recommendations | Potentially high | High | Vector matching is relevant, but this would introduce a new Python service into a mature PHP system |

## SimpliXio integration shape

```text
signal capture/update
        |
        +--> SimpliXio source-of-truth + deterministic scores
        |
        +--> local embedding --> ToucanDB upsert(signal_id)
                                  |
new signal/query -----------------+--> top semantic candidates
                                           |
                                           +--> recurrence evidence
                                           +--> related-signal edges
                                           +--> resurfacing explanation
```

### Apple application runtime

The packaged iOS and macOS apps use a native ToucanDB runtime rather than
embedding this Python distribution. The native implementation keeps ToucanDB's
important boundary—stable-ID upserts, content-aware reindexing, persistent
`Float32` vectors, deletion pruning, exact similarity search, and deterministic
fallback—while mapping it to platform components:

- Apple Natural Language supplies on-device sentence embeddings;
- SQLite WAL stores the rebuildable vector index;
- Accelerate performs normalized matrix-vector search; and
- Swift actors serialize mutations and keep UI queries responsive.

This avoids packaging Python, NumPy, SciPy, PyTorch, or FAISS into the app and
does not require a local HTTP service. The Python `SimplixioSignalMemory`
adapter remains appropriate for backend ingestion, evaluation, and larger
server-side corpora. Devices synchronize source records, not model-specific
vectors, and rebuild their own index locally.

Use semantic similarity as candidate generation, not as an opaque final score.
A safe rollout is:

1. backfill `signal_records.json` into ToucanDB;
2. log Jaccard and semantic candidates side by side without changing queues;
3. inspect false positives, especially across sensitive/project boundaries;
4. require matching project/sensitivity constraints before accepting evidence;
5. blend a calibrated semantic score into recurrence only after evaluation; and
6. keep Jaccard as a deterministic fallback if the model or index is unavailable.

## Run the working adapter

Install the local embedding extra:

```bash
pip install -e '.[embeddings]'
```

Set an encryption key and run the example against a SimpliXio data directory:

```bash
export TOUCANDB_ENCRYPTION_KEY='use-a-secret-from-your-keychain'
python examples/simplixio_signal_memory.py \
  --signals ../CortexOSLLM/.cortexos_local/signal_records.json \
  --query 'What did I previously decide about the desktop beta?' \
  --project SimpliXio
```

The first run downloads the selected Sentence Transformers model. Later runs
reuse the local model and idempotently update the signal index.

## Acceptance criteria for the product integration

- Re-syncing the same records does not increase the logical vector count.
- Editing a signal replaces its search document and metadata.
- Removing a signal from a complete snapshot removes it from semantic results.
- Project filtering never leaks a result from another project.
- Semantic candidates recover a reviewed paraphrase set that Jaccard misses.
- SimpliXio still produces its normal queues when ToucanDB is absent.
- Sensitive signal text stays on-device when using the local provider, and the
  ToucanDB store is encrypted with a key managed outside the repository.

## ToucanDB changes justified by this use case

- stable development embeddings across process restarts;
- a lazy, local `SentenceTransformerEmbeddingProvider`;
- automatic document-collection creation from provider dimensions;
- explicit vector and document upserts;
- model-aware no-op upserts so unchanged signals are not re-embedded;
- SQLite-atomic bulk pruning and tombstone compaction;
- complete metadata-filter candidate evaluation; and
- the `SimplixioSignalMemory` adapter and executable example.

These are generic capabilities. They also unlock the PDF, VideoCards, learning,
and memory-retrieval candidates without turning ToucanDB into application
business logic.
