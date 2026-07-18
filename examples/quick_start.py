#!/usr/bin/env python3
"""Minimal raw-vector ToucanDB example with safe repeatable upserts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from toucandb import SearchQuery, ToucanDB, create_schema


async def main() -> None:
    database_path = Path("./example_db")
    encryption_key = os.environ.get("TOUCANDB_ENCRYPTION_KEY")
    async with await ToucanDB.create(
        database_path,
        encryption_key=encryption_key,
    ) as db:
        if "birds" not in db.list_collections():
            await db.create_collection(
                create_schema("birds", dimensions=3, index_type="hnsw")
            )

        result = await db.upsert_vectors(
            "birds",
            [
                {
                    "id": "toucan",
                    "vector": [0.95, 0.10, 0.20],
                    "metadata": {"family": "Ramphastidae", "region": "Americas"},
                },
                {
                    "id": "macaw",
                    "vector": [0.80, 0.25, 0.15],
                    "metadata": {"family": "Psittacidae", "region": "Americas"},
                },
                {
                    "id": "puffin",
                    "vector": [0.10, 0.90, 0.40],
                    "metadata": {"family": "Alcidae", "region": "Atlantic"},
                },
            ],
        )
        if not result.success:
            raise RuntimeError(result.error_message)

        matches = await db.search_vectors(
            "birds",
            SearchQuery(
                vector=[1.0, 0.0, 0.0],
                k=2,
                include_metadata=True,
                metadata_filter={"region": "Americas"},
            ),
        )
        if not matches.success:
            raise RuntimeError(matches.error_message)

        for match in matches.data or []:
            print(f"{match['id']}: {match['score']:.3f} {match['metadata']}")


if __name__ == "__main__":
    asyncio.run(main())
