#!/usr/bin/env python3
"""Local semantic search with a lazily loaded Sentence Transformers model."""

from __future__ import annotations

import asyncio
import os

from toucandb import SentenceTransformerEmbeddingProvider, ToucanDB


async def main() -> None:
    documents = [
        "ToucanDB keeps durable vector records in SQLite WAL.",
        "FAISS accelerates nearest-neighbour search over embeddings.",
        "A stale index snapshot is rebuilt from the SQLite source of truth.",
        "SimpliXio uses semantic candidates alongside deterministic ranking.",
    ]
    provider = SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2")
    async with await ToucanDB.create(
        "./semantic_search_db",
        encryption_key=os.environ.get("TOUCANDB_ENCRYPTION_KEY"),
    ) as db:
        db.set_embedding_provider(provider)
        await db.ensure_document_collection("documentation")
        result = await db.upsert_documents(
            "documentation",
            documents,
            ids=[f"doc-{index}" for index in range(len(documents))],
        )
        if not result.success:
            raise RuntimeError(result.error_message)

        matches = await db.semantic_search(
            "documentation",
            "How does ToucanDB recover an invalid search index?",
            k=3,
        )
        for match in matches:
            print(f"{match['score']:.3f}: {match['document']}")


if __name__ == "__main__":
    asyncio.run(main())
