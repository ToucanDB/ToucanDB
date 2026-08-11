# ToucanDB 2.0.0 release notes

ToucanDB 2.0 turns the project into a serverless embedded vector engine with an
atomic SQLite source of truth, rebuildable FAISS snapshots, bounded caching,
and a framework-neutral RAG pipeline.

Highlights:

- complete crash recovery and backup behavior;
- stable, model-aware document synchronization that skips unchanged embeddings;
- local Sentence Transformers, OpenAI, and custom embedding adapters;
- first-class RAG retrieval and a tested SimpliXio semantic-memory adapter;
- privacy-conscious pipeline traces, retrieval evaluation, and optional FAISS
  semantic topic discovery;
- materially smaller core dependency graph;
- Python 3.10–3.14 support; and
- explicit single-owner, security, performance, Apple, and migration guidance.

This is a major release. Read [the migration guide](docs/migration-2.0.md)
before opening a 1.x database. Keep a filesystem backup and the encryption key.

Created and maintained by **Pierre-Henry Soria** —
[website](https://pierrehenry.dev), [GitHub](https://github.com/pH-7), and
[LinkedIn](https://www.linkedin.com/in/ph7enry/).
