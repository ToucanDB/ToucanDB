# Architecture

ToucanDB 2 is an embedded, single-owner vector engine. The Python process that
opens a database owns its durable store and in-memory search indices until it
closes them.

## Components

Each collection has two deliberately separate layers:

1. SQLite in WAL mode is the source of truth. A vector, its metadata, ID, and
   timestamp occupy one row, so an insert or replacement cannot leave a vector
   and metadata pair half-written.
2. FAISS is a rebuildable search accelerator. It holds normalized FP32 vectors
   for cosine search and the selected flat, HNSW, or IVF structure.

The database-wide cache budget is divided across open collections. Each
collection evicts least-recently-used records by estimated bytes. A zero-megabyte
budget disables the record cache; FAISS still owns its search index memory.

## Open and recovery sequence

```text
acquire exclusive database lock
  -> load and validate schemas
  -> open each SQLite collection
  -> verify encryption marker
  -> compare storage generation with snapshot manifest
       -> match: load FAISS snapshot and ID map
       -> mismatch/read failure: stream SQLite rows and rebuild
```

The manifest also contains a schema hash. A snapshot copied from another CPU
architecture may not be readable by FAISS; that read failure is safe because
SQLite records are portable and rebuild the index.

## Write consistency

An insert/upsert is validated before storage work. The changed batch commits in
one SQLite transaction, which increments the storage generation once. ToucanDB
then updates FAISS. If the index update fails, it immediately rebuilds from the
committed source of truth.

Deletes use the same ordering. HNSW deletion is represented as a logical
tombstone because physical removal is expensive. ToucanDB searches enough
physical candidates to compensate, then rebuilds when the configured count and
ratio thresholds are both reached. Flat and IVF use the same consistent policy.

Index snapshots are written to generation-specific files. The manifest is
atomically replaced last, so an interrupted snapshot never points to a partial
pair. Old snapshot files are removed only after the new manifest is visible.

## Storage layout

```text
database/
├── .toucandb.lock
├── encryption.salt             # only when encryption is configured
├── schemas/
│   └── collection.json
└── collection/
    ├── vectors.sqlite3
    ├── vectors.sqlite3-wal     # present while needed by SQLite
    └── index/
        ├── manifest.json
        ├── index-<generation>-<token>.faiss
        └── mapping-<generation>-<token>.msgpack
```

Vector IDs are serialized with an explicit string/integer type and hashed for
the SQLite primary key. IDs never become paths, and the string `"7"` remains
distinct from the integer `7`.

## Concurrency model

- Async public operations run blocking index, SQLite, and local-model work in
  worker threads so the caller's event loop remains responsive.
- A collection lock keeps its SQLite and FAISS views consistent.
- Independent collections can do work concurrently.
- An OS-level lock permits one ToucanDB owner for a database directory. This is
  required because separate processes would otherwise have separate FAISS
  state.

For multiple application workers or devices, put one ToucanDB-owning process
behind an application API. Do not place a live database directory on a network
filesystem or share it between containers.

## Backup

`await db.backup(path)` uses SQLite's online backup API while collection locks
are held, snapshots each index, copies schemas and the database salt, and then
atomically exposes the completed backup directory. A destination must be absent
or empty and outside the live database.

The backup can be opened directly with `ToucanDB.create(backup_path, ...)`. Keep
the encryption key separately; it is intentionally not part of the backup.

## Embedded boundary

ToucanDB does not include an HTTP server, identity layer, distributed consensus,
or cross-node replication. Those belong to the host application when needed.
This keeps a local deployment small and avoids consuming network, connection,
and worker resources that an embedded workload does not need.
