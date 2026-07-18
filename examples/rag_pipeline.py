#!/usr/bin/env python3
"""Runnable retrieval and context-building example without an LLM dependency."""

from __future__ import annotations

import asyncio
import os

from toucandb import SentenceTransformerEmbeddingProvider
from toucandb.integrations import RAGDocument, RAGStore


async def main() -> None:
    rag = await RAGStore.create(
        "./rag_example_db",
        SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2"),
        encryption_key=os.environ.get("TOUCANDB_ENCRYPTION_KEY"),
        namespace="toucan-docs",
    )
    try:
        synced = await rag.sync_documents(
            [
                RAGDocument(
                    id="architecture",
                    text=(
                        "SQLite is ToucanDB's durable source of truth. FAISS is a "
                        "rebuildable search accelerator whose snapshot is checked "
                        "against the SQLite generation."
                    ),
                    source="architecture.md",
                ),
                RAGDocument(
                    id="deployment",
                    text=(
                        "ToucanDB is embedded. One process owns a database directory; "
                        "shared deployments put that owner behind an application API."
                    ),
                    source="deployment.md",
                ),
            ],
            prune=True,
        )
        if not synced.success:
            raise RuntimeError(synced.error_message)

        hits = await rag.retrieve("What happens when an index snapshot is stale?", k=3)
        print(rag.format_context(hits))
    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
