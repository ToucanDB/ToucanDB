#!/usr/bin/env python3
"""Index SimpliXio signals and find semantically related prior signals.

Example:
    python examples/simplixio_signal_memory.py \
      --signals ../CortexOSLLM/.cortexos_local/signal_records.json \
      --query "What did I previously say about preparing a beta release?"
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from toucandb import SentenceTransformerEmbeddingProvider
from toucandb.integrations import SimplixioSignalMemory


async def run(args: argparse.Namespace) -> int:
    provider = SentenceTransformerEmbeddingProvider(args.model)
    encryption_key = os.environ.get(args.encryption_key_env)
    memory = await SimplixioSignalMemory.create(
        args.db,
        provider,
        encryption_key=encryption_key,
    )
    try:
        result = await memory.sync_file(args.signals)
        if not result.success:
            print(f"Signal synchronization failed: {result.error_message}")
            return 1

        matches = await memory.related(
            args.query,
            k=args.limit,
            project=args.project,
            min_score=args.min_score,
        )
        print(f"Indexed {len(result.data or [])} signals; found {len(matches)} matches")
        for position, match in enumerate(matches, 1):
            metadata = match["metadata"]
            print(
                f"{position}. {match['score']:.3f} [{metadata['signal_type']}] "
                f"{match['document'].splitlines()[0]}"
            )
        return 0
    finally:
        await memory.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use ToucanDB as SimpliXio's local semantic signal memory."
    )
    parser.add_argument(
        "--signals",
        type=Path,
        required=True,
        help="Path to SimpliXio's signal_records.json",
    )
    parser.add_argument(
        "--query", required=True, help="Natural-language signal to compare"
    )
    parser.add_argument("--db", type=Path, default=Path(".toucandb/simplixio-signals"))
    parser.add_argument("--project", help="Only return signals from this project")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--encryption-key-env",
        default="TOUCANDB_ENCRYPTION_KEY",
        help="Environment variable containing the at-rest encryption key",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
