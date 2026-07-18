# iOS and macOS integration

The Python distribution is not packaged into the SimpliXio iOS/macOS binaries.
Bundling Python, PyTorch, NumPy, and FAISS would increase app size, complicate
signing, and consume resources that Apple platforms already provide natively.

The native ToucanDB boundary used by SimpliXio maps the same semantics to:

- Apple Natural Language for on-device sentence embeddings;
- SQLite WAL for durable, rebuildable vector records;
- Accelerate for normalized matrix-vector similarity; and
- Swift actors for serialized mutation and responsive UI queries.

The application uses stable-ID upserts, content/model-aware reindexing,
deletion pruning, deterministic fallback, and source-record ownership. It does
not synchronize model-specific vectors between platforms. A device indexes its
own source records with the model available on that platform.

## When no backend is needed

An entirely local experience is appropriate when records stay on one device,
the Apple embedding model is sufficient, and platform data protection/keychain
meet the security requirements. No local HTTP server or embedded Python runtime
is needed.

## When a backend is useful

Use one application service when devices need shared records, server-controlled
authorization, larger server-side corpora, centralized evaluation, or hosted
model credentials. The service can embed ToucanDB's Python package. Synchronize
source records and provenance, then rebuild each platform index; do not treat
embedding vectors as the cross-model source of truth.

## Product integration rule

Semantic similarity should generate candidates. SimpliXio's deterministic
ranking, sensitivity boundaries, feedback, and fallback behavior remain the
product source of truth. Roll out by comparing deterministic and semantic
candidates, reviewing false positives, and calibrating before changing ranking.
