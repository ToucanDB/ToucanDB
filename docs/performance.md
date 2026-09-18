# Performance and tuning

ToucanDB optimizes for an embedded process with predictable resource ceilings.
There is no universal best index or published one-number performance claim.

## Start with the workload

Record at least:

- active vectors and expected growth;
- dimensions and vector dtype;
- query `k` and concurrency;
- filter selectivity;
- required recall against exact search;
- write/upsert/delete rate;
- cold-open and warm-query requirements; and
- target hardware, Python, NumPy, and FAISS versions.

## Index choice

Use flat search as the correctness baseline. It is often the right choice for a
small collection and is how approximate recall should be measured.

HNSW is the default for interactive retrieval. `hnsw_m` trades memory and build
cost for graph connectivity. `hnsw_ef_construction` affects build quality, while
query `ef` trades latency for recall. Tune on held-out queries.

IVF suits larger, batch-oriented corpora. ToucanDB sizes the initial centroid
count conservatively from available training data. An IVF schema uses exact flat
search until enough training examples exist, then promotes automatically and
retrains when the useful centroid count doubles. Query `nprobe` trades work for
recall. Metadata-filter exhaustion probes all IVF lists so correctness is not
silently limited by the default.

## Writes and synchronization

- Send batches rather than one record per call. SQLite commits the changed
  vectors in one transaction.
- Use stable IDs and upserts for synchronization.
- Document upserts compare stored text, metadata, and embedding-provider identity
  before calling the model.
- Use `delete_vectors()` for bulk removal. RAG and SimpliXio pruning already do.
- Call `await collection.optimize()` after a large one-time ingest if the final
  IVF training distribution matters or an immediate compact snapshot is wanted.

Replacements and deletes create logical tombstones. FAISS skips them during the
search through an ID selector, so they do not inflate the number of results
requested; a flat scan with few tombstones over-fetches instead, because a
selector would cost it the BLAS kernel. Automatic compaction requires both the
configured minimum tombstone count and ratio, avoiding rebuild churn for a few
edits. Tune these through `DatabaseConfig.PerformanceConfig`.

Tombstones still cost HNSW recall when they are clustered. Deleting most of a
query's neighbourhood, as pruning one topic's documents does, leaves the graph
walk spending its `ef` budget on dead nodes. Raise `ef` for those queries, lower
the compaction thresholds, or call `await collection.optimize()` after a large
prune.

## Metadata filters

Filters are applied after the vector search. ToucanDB widens the candidate set
until `k` records match, reading candidates in batched queries sized by the
observed match rate. When an approximate index cannot supply more candidates,
one exact pass guarantees that every matching vector is considered. A filter
matching very few vectors therefore costs roughly an exact scan plus a read of
the candidates ranked ahead of the matches; keep such filters for small
collections or partition the data into separate collections.

## Memory

FAISS owns the search index. A rough lower bound for stored FP32 vector values
is `physical_vectors × dimensions × 4` bytes; HNSW graph structures and FAISS
overhead are additional.

`DatabaseConfig.MemoryConfig(cache_size_mb=64)` sets a total record-cache
ceiling shared across collections. It is not the FAISS limit. Set it to zero to
disable caching of result records. Local embedding models have their own memory
use and load only when first needed.

## Storage and durability

SQLite WAL with `synchronous=NORMAL` is used for a durable, high-throughput local
store. Each logical record is atomic. Encryption and compression add CPU work;
measure them in the intended configuration rather than benchmarking plaintext
and extrapolating.

Index snapshots improve clean reopen time. They are accelerators, not backups.
A missing, stale, invalid, or architecture-incompatible snapshot rebuilds from
SQLite.

## Reproducible benchmark

```bash
python benchmarks/benchmark.py \
  --vectors 10000 \
  --dimensions 384 \
  --queries 100 \
  --index hnsw \
  --seed 42
```

The script reports its configuration, Python/platform versions, ingestion
throughput, latency percentiles, database size, and process RSS when `psutil` is
installed. It uses synthetic normalized vectors and is a regression tool—not a
claim about semantic recall or a production corpus.

For approximate indices, add a workload-specific harness that compares result
IDs with a flat index. Always publish warm/cold state, vector distribution,
hardware, filter distribution, encryption, and tuning values alongside results.
