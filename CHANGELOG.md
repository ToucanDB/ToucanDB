# Changelog

All notable changes to ToucanDB are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-09-12

### Added

- SQLite WAL source-of-truth storage with atomic batch upsert/delete.
- Generation- and schema-checked FAISS snapshots with crash-safe rebuild.
- Database-wide bounded LRU cache budgets.
- Automatic HNSW tombstone compaction and adaptive IVF sizing/retraining.
- Complete, directly openable online backups.
- Cross-process single-owner locking.
- Dependency-free `RAGStore`, deterministic chunking, namespace pruning,
  attributed retrieval, bounded context, and generator orchestration.
- Observable `RAGPipeline` stage traces, optional postprocessing, reviewed
  retrieval metrics, and bounded namespace-isolated FAISS topic discovery.
- Lazy Sentence Transformers, OpenAI, and callable embedding providers.
- Embedding/model-aware no-op document upserts.
- Bulk vector deletion and SimpliXio bulk pruning.
- Typed-package marker and Python 3.10–3.14 CI coverage.
- Architecture, RAG, performance, Apple, migration, credits, and security docs.

### Fixed

- Collection drop now removes storage instead of only hiding its schema.
- Backup no longer reports success after copying schemas alone.
- Vector IDs cannot escape the database directory or collide by stringified
  type.
- INT8 records preserve the scale needed for reconstruction.
- Small IVF collections no longer fail training when `nlist` exceeds examples.
- Dot-product flat/HNSW/IVF indices use inner-product distance correctly.
- Metadata filtering expands candidates adaptively and exhausts IVF probes when
  necessary.
- Duplicate IDs, invalid metadata, non-finite vectors, limits, and dimensions
  are validated before a multi-batch commit.
- Importing ToucanDB no longer configures the host application's root logger.
- Blocking storage, FAISS, and local model calls no longer run on the event loop.
- Collection-load failures are no longer silently logged and skipped.

### Changed

- Python 3.10 is now the minimum supported version.
- Core dependencies are limited to NumPy, FAISS CPU, Cryptography, Pydantic,
  MessagePack, and LZ4.
- Sentence Transformers and OpenAI are separate optional extras.
- Unsupported metrics, indices, compression, and quantization modes fail at
  schema creation rather than later at runtime.
- `drop_collection(overwrite=True)` and storage configuration semantics are
  explicit and destructive only when requested.
- The development-status classifier is Beta to match the release's maturity.

### Removed

- Unused SciPy, xxhash, aiofiles, and uvloop core dependencies.
- Redundant direct Transformers, Torch, Cohere, LangChain, and Datasets extras.
- The non-functional FAISS GPU extra.
- Configuration fields that did not affect runtime behavior.

### Migration

See [docs/migration-2.0.md](docs/migration-2.0.md). Legacy file-per-vector
collections migrate automatically and their original files are preserved.

## [1.0.0] - 2026-03-20

- Initial PyPI release.

[2.0.0]: https://github.com/ToucanDB/ToucanDB/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/ToucanDB/ToucanDB/releases/tag/v1.0.0
