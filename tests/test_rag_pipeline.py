"""Observable RAG pipeline, evaluation, and FAISS discovery tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from toucandb.integrations import (
    RAGDocument,
    RAGPipeline,
    RAGPipelineTrace,
    RAGStore,
    RetrievalEvaluationCase,
    TextChunker,
)


class TopicEmbeddingProvider:
    """Small deterministic semantic provider for pipeline tests."""

    dimensions = 3
    model_id = "topic-pipeline-v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            values = [
                float(sum(term in lowered for term in ("release", "beta", "ship"))),
                float(sum(term in lowered for term in ("database", "vector", "rag"))),
                float(sum(term in lowered for term in ("toucan", "bird", "forest"))),
            ]
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            embeddings.append([value / norm for value in values])
        return embeddings


async def create_pipeline(path: Path) -> tuple[RAGStore, RAGPipeline]:
    rag = await RAGStore.create(
        path,
        TopicEmbeddingProvider(),
        namespace="pipeline-tests",
        chunker=TextChunker(chunk_size=256, chunk_overlap=24),
    )
    synced = await rag.sync_documents(
        [
            RAGDocument(
                id="release",
                text="Ship the beta release after the final review.",
                source="release.md",
            ),
            RAGDocument(
                id="database",
                text="ToucanDB is a vector database for local RAG.",
                source="database.md",
            ),
            RAGDocument(
                id="bird",
                text="The toucan bird lives near the forest canopy.",
                source="bird.md",
            ),
        ],
        prune=True,
    )
    assert synced.success
    return rag, RAGPipeline(rag)


@pytest.mark.asyncio
async def test_pipeline_traces_stages_without_recording_content(tmp_path: Path) -> None:
    rag, _ = await create_pipeline(tmp_path / "observable")
    observed: list[RAGPipelineTrace] = []

    async def observer(trace: RAGPipelineTrace) -> None:
        observed.append(trace)
        raise RuntimeError("monitoring is temporarily unavailable")

    pipeline = RAGPipeline(rag, observer=observer)

    async def generator(prompt: str) -> str:
        assert "database.md" in prompt
        return "ToucanDB supports local RAG [1]."

    result = await pipeline.run(
        "Which vector database supports local RAG?",
        generator,
        k=2,
        postprocessor=lambda answer: answer.replace("supports", "powers"),
        trace_id="trace-for-test",
    )

    assert result.answer.answer == "ToucanDB powers local RAG [1]."
    assert result.trace.trace_id == "trace-for-test"
    assert result.trace.success
    assert result.trace.embedding_provider == "topic-pipeline-v1"
    assert [stage.name for stage in result.trace.stages] == [
        "retrieve",
        "context",
        "prompt",
        "generate",
        "postprocess",
    ]
    assert all(stage.duration_ms >= 0 for stage in result.trace.stages)
    assert result.trace.context_characters == len(result.answer.context)
    assert result.trace.output_characters == len(result.answer.answer)
    assert "Which vector database" not in repr(result.trace)
    assert observed == [result.trace]
    await rag.close()


@pytest.mark.asyncio
async def test_pipeline_reports_failures_without_masking_them(tmp_path: Path) -> None:
    rag, _ = await create_pipeline(tmp_path / "failure")
    observed: list[RAGPipelineTrace] = []
    pipeline = RAGPipeline(rag, observer=observed.append)

    with pytest.raises(TypeError, match="generator must return a string"):
        await pipeline.run("Which database supports RAG?", lambda _: 42)  # type: ignore[arg-type]

    assert len(observed) == 1
    assert not observed[0].success
    assert observed[0].error_type == "TypeError"
    assert observed[0].retrieved_ids
    await rag.close()


@pytest.mark.asyncio
async def test_retrieval_evaluation_separates_search_quality_from_generation(
    tmp_path: Path,
) -> None:
    rag, pipeline = await create_pipeline(tmp_path / "evaluation")
    report = await pipeline.evaluate_retrieval(
        [
            RetrievalEvaluationCase(
                query="prepare to ship the beta",
                relevant_document_ids=frozenset({"release"}),
            ),
            RetrievalEvaluationCase(
                query="vector database for retrieval augmented generation",
                relevant_document_ids=frozenset({"database"}),
            ),
        ],
        k=1,
        concurrency=2,
    )

    assert report.case_count == 2
    assert report.hit_rate_at_k == 1.0
    assert report.mean_recall_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0
    assert report.average_latency_ms >= 0
    assert report.p95_latency_ms >= 0
    assert all(result.relevant_ranks == (1,) for result in report.results)
    await rag.close()


@pytest.mark.asyncio
async def test_faiss_topic_discovery_is_bounded_and_namespace_isolated(
    tmp_path: Path,
) -> None:
    rag = await RAGStore.create(
        tmp_path / "clusters",
        TopicEmbeddingProvider(),
        namespace="visible",
        chunker=TextChunker(chunk_size=256, chunk_overlap=24),
    )
    groups = {
        "release": {
            "release-a": "Ship the beta release.",
            "release-b": "Prepare the release for beta shipping.",
            "release-c": "Review the beta before release.",
        },
        "database": {
            "database-a": "A vector database powers RAG.",
            "database-b": "Use the database for vector retrieval.",
            "database-c": "RAG searches the vector database.",
        },
        "bird": {
            "bird-a": "A toucan bird rests in the forest.",
            "bird-b": "The forest is home to the toucan bird.",
            "bird-c": "Watch a bird near the forest toucan.",
        },
    }
    documents = [
        RAGDocument(id=document_id, text=text, source=f"{document_id}.md")
        for group in groups.values()
        for document_id, text in group.items()
    ]
    assert (await rag.sync_documents(documents, prune=True)).success

    hidden = RAGStore(rag.db, namespace="hidden")
    assert (
        await hidden.sync_documents(
            [
                RAGDocument(id="secret-a", text="Ship a private beta release."),
                RAGDocument(id="secret-b", text="Private release shipping notes."),
            ],
            prune=True,
        )
    ).success

    pipeline = RAGPipeline(rag)
    report = await pipeline.discover_clusters(
        3,
        representative_count=2,
        seed=7,
    )

    assert report.algorithm == "faiss-spherical-kmeans"
    assert report.vector_count == 9
    assert report.cluster_count == 3
    assert sum(cluster.member_count for cluster in report.clusters) == 9
    assert all(cluster.representative_hits for cluster in report.clusters)
    cluster_documents = [set(cluster.document_ids) for cluster in report.clusters]
    for group in groups.values():
        assert set(group).issubset(
            next(ids for ids in cluster_documents if set(group) <= ids)
        )
    assert all("secret-a" not in ids for ids in cluster_documents)

    with pytest.raises(ValueError, match="max_vectors=8"):
        await pipeline.discover_clusters(3, max_vectors=8)
    with pytest.raises(ValueError, match="max_memory_bytes=1"):
        await pipeline.discover_clusters(3, max_memory_bytes=1)
    await rag.close()
