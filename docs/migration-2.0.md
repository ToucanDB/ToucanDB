# Migrating to ToucanDB 2.0

ToucanDB 2 is a major release because it changes the persistence engine,
supported Python floor, configuration surface, and validation behavior.

## Before upgrading

1. Stop every process that can access the database.
2. Make a filesystem copy of the entire 1.x database directory. The 1.x
   `backup()` method copied schemas only and is not a complete backup.
3. Record the encryption key outside the backup.
4. Upgrade and open a copied database first.

## Requirements and packaging

- Python 3.10+ replaces Python 3.9 support.
- The core no longer installs SciPy, xxhash, aiofiles, or uvloop because ToucanDB
  did not use them.
- `toucandb[embeddings]` installs Sentence Transformers. `toucandb[ml]` remains
  an alias for the 2.x transition.
- `toucandb[openai]` installs the OpenAI adapter.
- The old GPU extra is removed because upstream FAISS no longer publishes the
  GPU wheel. A custom FAISS build must be managed by the deployment.

## Automatic data migration

On first open, each 1.x collection's `vectors/*.vec` and `metadata/*.meta` pairs
are decoded, validated, and imported into `vectors.sqlite3`. Encrypted legacy
records use the supplied 1.x key. The original files are left untouched as a
rollback aid, and a migration marker prevents repeated imports.

The new FAISS snapshot is built from migrated SQLite rows. Do not delete the 1.x
backup until the application has verified vector counts and representative
searches.

## Behavioral changes

- One process/instance may own a database directory. A second open fails.
- Collection names are portable path-safe identifiers: 1–128 letters, numbers,
  dots, underscores, or hyphens, beginning with a letter or number.
- Vector IDs accept non-empty strings and non-boolean integers. They no longer
  become filenames, so slashes and other characters are safe.
- Duplicate IDs, non-finite values, non-serializable metadata, unsupported
  schema choices, and collection limits fail before a multi-batch write starts.
- `drop_collection()` now removes collection storage and its schema. In 1.x it
  reported success but left files behind.
- `backup()` now creates a complete, directly openable backup and rejects a
  non-empty destination.
- INT8 persistence now stores its quantization scale. Existing 1.x INT8 records
  did not preserve enough information for faithful reconstruction; re-embed or
  re-import them from original vectors.
- Dot-product thresholds are no longer incorrectly restricted to `[0, 1]`.

## Configuration changes

Configuration fields that had no implementation were removed. The supported
resource knobs are:

```python
from toucandb import DatabaseConfig

config = DatabaseConfig(
    storage=DatabaseConfig.StorageConfig(path="./data"),
    memory=DatabaseConfig.MemoryConfig(cache_size_mb=64),
    performance=DatabaseConfig.PerformanceConfig(
        index_snapshot_interval=10_000,
        compaction_min_tombstones=64,
        compaction_ratio=0.20,
    ),
)
```

An encryption key passed to `ToucanDB.create()` enables encryption. A new key
cannot be applied in place to a 2.x plaintext collection; export source records
and import them into a newly encrypted database. This prevents a directory from
silently containing a mixture of plaintext and encrypted records.
