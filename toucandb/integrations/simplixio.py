"""Semantic signal memory for SimpliXio/CortexOS.

SimpliXio keeps deterministic ranking as its source of truth. This adapter adds
semantic candidate retrieval for recurrence and relationship evidence, solving
the cases where token-overlap matching misses a paraphrased signal.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .. import ToucanDB
from ..ml import EmbeddingProvider
from ..types import ErrorCode, OperationResult, VectorId


class SimplixioSignalMemory:
    """Synchronize and semantically query SimpliXio signal records."""

    DEFAULT_COLLECTION = "simplixio_signals"

    def __init__(
        self,
        db: ToucanDB,
        *,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.db = db
        self.collection_name = collection_name

    @classmethod
    async def create(
        cls,
        storage_path: str | Path,
        embedding_provider: EmbeddingProvider,
        *,
        encryption_key: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> SimplixioSignalMemory:
        db = await ToucanDB.create(storage_path, encryption_key=encryption_key)
        db.set_embedding_provider(embedding_provider)
        await db.ensure_document_collection(
            collection_name,
            metadata_schema={
                "signal_type": "string",
                "project": "string",
                "sensitivity": "string",
                "captured_at": "string",
                "source": "string",
            },
        )
        return cls(db, collection_name=collection_name)

    @staticmethod
    def _document(record: dict[str, Any]) -> str:
        text = str(record.get("text", "")).strip()
        topics = " ".join(str(value) for value in record.get("topics", []))
        tags = " ".join(str(value) for value in record.get("tags", []))
        projects = " ".join(str(value) for value in record.get("linked_projects", []))
        return "\n".join(
            value
            for value in (
                text,
                f"Topics: {topics}" if topics else "",
                f"Tags: {tags}" if tags else "",
                f"Projects: {projects}" if projects else "",
            )
            if value
        )

    @staticmethod
    def _metadata(record: dict[str, Any]) -> dict[str, Any]:
        projects = record.get("linked_projects", [])
        return {
            "signal_type": str(record.get("signal_type", "thought")),
            "project": str(projects[0]) if projects else "",
            "sensitivity": str(record.get("sensitivity", "sensitive")),
            "captured_at": str(record.get("captured_at", "")),
            "source": str(record.get("source", "")),
            "topics": [str(value) for value in record.get("topics", [])],
            "tags": [str(value) for value in record.get("tags", [])],
        }

    async def sync_records(
        self, records: list[dict[str, Any]], *, prune: bool = False
    ) -> OperationResult[list[VectorId]]:
        """Idempotently index current signal records by their stable IDs.

        Set ``prune`` when the input is a complete source-of-truth snapshot to
        remove records that SimpliXio has deleted.
        """
        valid_records = [
            record
            for record in records
            if str(record.get("id", "")).strip() and str(record.get("text", "")).strip()
        ]
        result = await self.db.upsert_documents(
            self.collection_name,
            [self._document(record) for record in valid_records],
            metadata=[self._metadata(record) for record in valid_records],
            ids=[str(record["id"]) for record in valid_records],
        )
        if result.success and prune:
            current_ids = {str(record["id"]) for record in valid_records}
            stale_ids = [
                vector_id
                for vector_id in self.db.list_vector_ids(self.collection_name)
                if str(vector_id) not in current_ids
            ]
            delete_result = await self.db.delete_vectors(
                self.collection_name,
                stale_ids,
            )
            if not delete_result.success:
                return OperationResult.error_result(
                    delete_result.error_code or ErrorCode.STORAGE_ERROR,
                    delete_result.error_message or "Signal prune failed",
                    delete_result.execution_time_ms,
                )
        return result

    async def sync_file(
        self, records_path: str | Path
    ) -> OperationResult[list[VectorId]]:
        """Load SimpliXio's ``signal_records.json`` and synchronize it."""
        payload = await asyncio.to_thread(
            lambda: json.loads(Path(records_path).read_text(encoding="utf-8"))
        )
        if not isinstance(payload, list):
            raise ValueError("SimpliXio signal records must be a JSON list")
        records = [record for record in payload if isinstance(record, dict)]
        return await self.sync_records(records, prune=True)

    async def related(
        self,
        text: str,
        *,
        k: int = 5,
        project: str | None = None,
        exclude_signal_id: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Find semantically related signals with optional project scoping."""
        metadata_filter = {"project": project} if project is not None else None
        # Fetch one additional item when excluding the query's own record.
        fetch_k = k + (1 if exclude_signal_id else 0)
        results = await self.db.semantic_search(
            self.collection_name,
            text,
            k=fetch_k,
            filter_metadata=metadata_filter,
        )
        return [
            result
            for result in results
            if str(result.get("id")) != exclude_signal_id
            and float(result.get("score") or 0.0) >= min_score
        ][:k]

    async def close(self) -> None:
        """Close the underlying ToucanDB instance."""
        await self.db.close()

    async def __aenter__(self) -> SimplixioSignalMemory:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()
