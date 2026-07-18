#!/usr/bin/env python3
"""Index UTF-8 files with the RAG integration and retrieve attributed chunks."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from toucandb import SentenceTransformerEmbeddingProvider
from toucandb.integrations import RAGStore


async def run(paths: list[Path], query: str) -> None:
    rag = await RAGStore.create(
        "./document_search_db",
        SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2"),
        encryption_key=os.environ.get("TOUCANDB_ENCRYPTION_KEY"),
        namespace="local-files",
    )
    try:
        result = await rag.sync_paths(paths, prune=True)
        if not result.success:
            raise RuntimeError(result.error_message)
        for hit in await rag.retrieve(query, k=5):
            print(f"{hit.score:.3f} {hit.source}\n{hit.text}\n")
    finally:
        await rag.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--query", required=True)
    arguments = parser.parse_args()
    asyncio.run(run(arguments.paths, arguments.query))


if __name__ == "__main__":
    main()
