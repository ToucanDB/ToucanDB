"""ML document APIs and the concrete SimpliXio signal-memory use case."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from toucandb import (
    CallableEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SearchQuery,
    SimpleEmbeddingProvider,
    ToucanDB,
    create_schema,
)
from toucandb.integrations import SimplixioSignalMemory


class IntentEmbeddingProvider:
    """Small deterministic provider used to test semantic workflow plumbing."""

    dimensions = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            release = sum(
                term in lowered
                for term in ("release", "testflight", "beta", "shipping", "changed")
            )
            learning = sum(
                term in lowered for term in ("learn", "flashcard", "study", "video")
            )
            customer = sum(
                term in lowered
                for term in ("customer", "interview", "feedback", "user")
            )
            vector = [float(release), float(learning), float(customer)]
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


@pytest.mark.asyncio
async def test_simple_provider_is_stable_across_instances():
    first = SimpleEmbeddingProvider(dimensions=8, seed=42)
    second = SimpleEmbeddingProvider(dimensions=8, seed=42)

    assert await first.embed(["persistent signal"]) == await second.embed(
        ["persistent signal"]
    )


@pytest.mark.asyncio
async def test_callable_provider_adapts_sync_and_async_functions():
    sync_provider = CallableEmbeddingProvider(
        lambda texts: [[float(len(text)), 1.0] for text in texts],
        dimensions=2,
        model_id="sync-v1",
    )

    async def async_embed(texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 2.0] for text in texts]

    async_provider = CallableEmbeddingProvider(
        async_embed,
        dimensions=2,
        model_id="async-v1",
    )

    assert await sync_provider.embed(["abc"]) == [[3.0, 1.0]]
    assert await async_provider.embed(["abc"]) == [[3.0, 2.0]]


@pytest.mark.asyncio
async def test_openai_provider_is_lazy_batched_and_restores_response_order():
    class EmbeddingsEndpoint:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def create(self, **arguments):
            self.calls.append(arguments)
            values = arguments["input"]
            data = [
                SimpleNamespace(index=index, embedding=[float(len(text)), 1.0])
                for index, text in enumerate(values)
            ]
            return SimpleNamespace(data=list(reversed(data)))

    endpoint = EmbeddingsEndpoint()
    client = SimpleNamespace(embeddings=endpoint)
    provider = OpenAIEmbeddingProvider(
        "text-embedding-3-small",
        dimensions=2,
        client=client,
        batch_size=2,
    )

    embedded = await provider.embed(["a", "four", "xyz"])

    assert embedded == [[1.0, 1.0], [4.0, 1.0], [3.0, 1.0]]
    assert len(endpoint.calls) == 2
    assert all(call["dimensions"] == 2 for call in endpoint.calls)


@pytest.mark.asyncio
async def test_document_upsert_replaces_existing_id_without_growing_count(tmp_path):
    db = await ToucanDB.create(tmp_path / "db")
    db.set_embedding_provider(IntentEmbeddingProvider())
    await db.ensure_document_collection("signals")

    first = await db.insert_documents(
        "signals",
        ["Ship the TestFlight release"],
        metadata=[{"version": 1}],
        ids=["signal-1"],
    )
    duplicate = await db.insert_documents(
        "signals",
        ["Ship a changed release"],
        metadata=[{"version": 2}],
        ids=["signal-1"],
    )
    replacement = await db.upsert_documents(
        "signals",
        ["Document what changed before beta submission"],
        metadata=[{"version": 2}],
        ids=["signal-1"],
    )

    assert first.success
    assert not duplicate.success
    assert replacement.success
    assert db.get_collection_stats("signals").total_vectors == 1
    results = await db.semantic_search("signals", "prepare beta notes", k=1)
    assert results[0]["id"] == "signal-1"
    assert results[0]["metadata"]["version"] == 2


@pytest.mark.asyncio
async def test_flat_dot_product_uses_inner_product_index(tmp_path):
    db = await ToucanDB.create(tmp_path / "dot-db")
    await db.create_collection(
        create_schema("dot", 2, metric="dot_product", index_type="flat")
    )
    inserted = await db.insert_vectors(
        "dot",
        [
            {"id": "aligned", "vector": [2.0, 0.0]},
            {"id": "orthogonal", "vector": [0.0, 3.0]},
        ],
    )
    result = await db.search_vectors("dot", SearchQuery(vector=[1.0, 0.0], k=2))

    assert inserted.success and result.success
    assert [item["id"] for item in result.data or []] == [
        "aligned",
        "orthogonal",
    ]
    assert (result.data or [])[0]["score"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_simplixio_memory_finds_paraphrased_signals_and_syncs_idempotently(
    tmp_path,
):
    records = [
        {
            "id": "release-notes",
            "text": "Ship the macOS release notes before TestFlight review",
            "signal_type": "task",
            "linked_projects": ["SimpliXio"],
            "topics": ["desktop"],
            "tags": ["public_safe"],
            "sensitivity": "public_safe",
        },
        {
            "id": "customer-call",
            "text": "Interview customers about priority clarity",
            "signal_type": "task",
            "linked_projects": ["SimpliXio"],
            "topics": ["feedback"],
        },
        {
            "id": "study-video",
            "text": "Turn the architecture video into study flashcards",
            "signal_type": "idea",
            "linked_projects": ["VideoCards"],
            "topics": ["learning"],
        },
    ]
    memory = await SimplixioSignalMemory.create(
        tmp_path / "signals-db", IntentEmbeddingProvider(), encryption_key="test-key"
    )

    first_sync = await memory.sync_records(records)
    second_sync = await memory.sync_records(records)
    related = await memory.related(
        "Document what changed in the desktop build prior to beta submission",
        project="SimpliXio",
        k=2,
    )

    assert first_sync.success and second_sync.success
    assert memory.db.get_collection_stats(memory.collection_name).total_vectors == 3
    collection = memory.db.get_collection(memory.collection_name)
    assert collection.index.index.ntotal == 3
    assert related[0]["id"] == "release-notes"
    assert all(result["metadata"]["project"] == "SimpliXio" for result in related)

    pruned = await memory.sync_records(records[:2], prune=True)
    assert pruned.success
    assert memory.db.get_collection_stats(memory.collection_name).total_vectors == 2
    assert "study-video" not in {
        str(vector_id)
        for vector_id in memory.db.list_vector_ids(memory.collection_name)
    }

    await memory.close()
