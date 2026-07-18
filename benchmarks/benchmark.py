#!/usr/bin/env python3
"""Reproducible synthetic ToucanDB regression benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from toucandb import SearchQuery, ToucanDB, create_schema


def percentile(values: list[float], percentage: float) -> float:
    return float(np.percentile(np.asarray(values), percentage))


async def benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    random = np.random.default_rng(arguments.seed)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if arguments.path is None:
        temporary = tempfile.TemporaryDirectory(prefix="toucandb-benchmark-")
        database_path = Path(temporary.name)
    else:
        database_path = arguments.path.expanduser().resolve()

    encryption_key = None
    if arguments.encrypt:
        encryption_key = os.environ.get("TOUCANDB_BENCHMARK_KEY")
        if not encryption_key:
            raise RuntimeError(
                "--encrypt requires TOUCANDB_BENCHMARK_KEY in the environment"
            )

    db = await ToucanDB.create(database_path, encryption_key=encryption_key)
    try:
        await db.create_collection(
            create_schema(
                "benchmark",
                arguments.dimensions,
                index_type=arguments.index,
            )
        )
        ingestion_start = time.perf_counter()
        inserted = 0
        while inserted < arguments.vectors:
            count = min(arguments.batch_size, arguments.vectors - inserted)
            matrix = random.normal(size=(count, arguments.dimensions)).astype(
                np.float32
            )
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            vectors = [
                {
                    "id": f"vector-{inserted + offset}",
                    "vector": matrix[offset].tolist(),
                    "metadata": {"partition": (inserted + offset) % 10},
                }
                for offset in range(count)
            ]
            result = await db.insert_vectors("benchmark", vectors)
            if not result.success:
                raise RuntimeError(result.error_message)
            inserted += count
        ingestion_seconds = time.perf_counter() - ingestion_start

        query_matrix = random.normal(
            size=(arguments.queries, arguments.dimensions)
        ).astype(np.float32)
        query_matrix /= np.maximum(
            np.linalg.norm(query_matrix, axis=1, keepdims=True), 1e-12
        )
        latencies: list[float] = []
        for query_vector in query_matrix:
            started = time.perf_counter()
            result = await db.search_vectors(
                "benchmark",
                SearchQuery(vector=query_vector, k=arguments.k),
            )
            if not result.success:
                raise RuntimeError(result.error_message)
            latencies.append((time.perf_counter() - started) * 1000)

        stats = db.get_collection_stats("benchmark")
        report: dict[str, Any] = {
            "configuration": {
                "vectors": arguments.vectors,
                "dimensions": arguments.dimensions,
                "queries": arguments.queries,
                "k": arguments.k,
                "index": arguments.index,
                "batch_size": arguments.batch_size,
                "seed": arguments.seed,
                "encrypted": arguments.encrypt,
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "numpy": np.__version__,
            },
            "ingestion": {
                "seconds": ingestion_seconds,
                "vectors_per_second": arguments.vectors / ingestion_seconds,
            },
            "search_ms": {
                "mean": statistics.fmean(latencies),
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "p99": percentile(latencies, 99),
                "max": max(latencies),
            },
            "database_size_bytes": stats.size_bytes,
        }
        try:
            import psutil  # type: ignore[import-not-found]

            report["process_rss_bytes"] = psutil.Process().memory_info().rss
        except ImportError:
            report["process_rss_bytes"] = None
        return report
    finally:
        await db.close()
        if temporary is not None:
            temporary.cleanup()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=10_000)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--index", choices=("flat", "hnsw", "ivf"), default="hnsw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--encrypt", action="store_true")
    arguments = parser.parse_args()
    for field in ("vectors", "dimensions", "queries", "k", "batch_size"):
        if getattr(arguments, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    return arguments


def main() -> None:
    report = asyncio.run(benchmark(parse_arguments()))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
