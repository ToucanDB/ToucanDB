"""Observable, evaluable, local-first RAG pipelines for ToucanDB.

This module deliberately stays framework-neutral. It adds production tracing,
retrieval evaluation, and optional FAISS topic discovery without requiring a
web server, orchestration framework, or hosted model.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import faiss  # type: ignore[import-not-found,import-untyped]
import numpy as np

from ..ml import embedding_provider_id
from ..types import DistanceMetric, VectorId
from .rag import (
    Generator,
    RAGAnswer,
    RAGHit,
    RAGStore,
    build_grounded_prompt,
    invoke_generator,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGStageTrace:
    """Timing and output count for one pipeline stage."""

    name: str
    duration_ms: float
    output_count: int | None = None


@dataclass(frozen=True)
class RAGPipelineTrace:
    """Privacy-conscious trace that excludes raw prompts, context, and answers."""

    trace_id: str
    started_at: str
    success: bool
    total_ms: float
    stages: tuple[RAGStageTrace, ...]
    embedding_provider: str
    retrieved_ids: tuple[VectorId, ...]
    retrieved_scores: tuple[float, ...]
    context_characters: int
    output_characters: int
    error_type: str | None = None


class RAGPipelineObserver(Protocol):
    """Receives completed success or failure traces for external monitoring."""

    def __call__(self, trace: RAGPipelineTrace) -> None | Awaitable[None]: ...


Postprocessor = Callable[[str], str | Awaitable[str]]


@dataclass(frozen=True)
class RAGPipelineResult:
    """Generated answer and its reproducible, content-free execution trace."""

    answer: RAGAnswer
    trace: RAGPipelineTrace


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """One reviewed query and the documents considered relevant to it."""

    query: str
    relevant_document_ids: frozenset[str]
    metadata_filter: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("evaluation query cannot be empty")
        if not self.relevant_document_ids:
            raise ValueError("relevant_document_ids cannot be empty")


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    """Per-case retrieval ranks and metrics."""

    query: str
    relevant_document_ids: frozenset[str]
    retrieved_document_ids: tuple[str, ...]
    relevant_ranks: tuple[int, ...]
    hit: bool
    recall_at_k: float
    reciprocal_rank: float
    latency_ms: float


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """Aggregate retrieval quality and latency, evaluated independently of an LLM."""

    k: int
    case_count: int
    hit_rate_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    average_latency_ms: float
    p95_latency_ms: float
    results: tuple[RetrievalEvaluationResult, ...]


@dataclass(frozen=True)
class RAGCluster:
    """One semantic neighborhood and its nearest representative chunks."""

    cluster_id: int
    member_count: int
    member_ids: tuple[VectorId, ...]
    document_ids: tuple[str, ...]
    representative_hits: tuple[RAGHit, ...]
    mean_distance: float


@dataclass(frozen=True)
class RAGClusteringReport:
    """Bounded, non-persistent FAISS clustering result for knowledge discovery."""

    algorithm: str
    vector_count: int
    cluster_count: int
    objective: float
    training_ms: float
    assignment_ms: float
    clusters: tuple[RAGCluster, ...]


@dataclass(frozen=True)
class _ClusterRecord:
    id: VectorId
    document_id: str
    text: str
    metadata: dict[str, Any]
    source: str | None
    vector: np.ndarray[Any, Any]


class RAGPipeline:
    """Orchestrate, trace, evaluate, and explore one :class:`RAGStore`."""

    def __init__(
        self,
        store: RAGStore,
        *,
        observer: RAGPipelineObserver | None = None,
    ):
        self.store = store
        self.observer = observer

    @property
    def _provider_id(self) -> str:
        provider = self.store.db.embedding_provider
        return embedding_provider_id(provider) if provider is not None else "none"

    async def _notify(self, trace: RAGPipelineTrace) -> None:
        if self.observer is None:
            return
        try:
            observed = self.observer(trace)
            if inspect.isawaitable(observed):
                await observed
        except Exception:
            # Monitoring must not turn a successful product request into a failure.
            logger.exception(
                "RAG pipeline observer failed for trace %s", trace.trace_id
            )

    @staticmethod
    async def _postprocess(postprocessor: Postprocessor, answer: str) -> str:
        processed = postprocessor(answer)
        result = await processed if inspect.isawaitable(processed) else processed
        if not isinstance(result, str):
            raise TypeError("RAG postprocessor must return a string")
        return result

    async def run(
        self,
        question: str,
        generator: Generator,
        *,
        k: int = 6,
        metadata_filter: dict[str, Any] | None = None,
        min_score: float | None = None,
        max_context_characters: int = 12_000,
        postprocessor: Postprocessor | None = None,
        trace_id: str | None = None,
    ) -> RAGPipelineResult:
        """Run a grounded answer pipeline and emit a privacy-safe stage trace."""
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        current_trace_id = trace_id or uuid.uuid4().hex
        stages: list[RAGStageTrace] = []
        hits: list[RAGHit] = []
        context = ""
        output = ""

        def record_stage(
            name: str, stage_started: float, output_count: int | None = None
        ) -> None:
            stages.append(
                RAGStageTrace(
                    name=name,
                    duration_ms=(time.perf_counter() - stage_started) * 1000,
                    output_count=output_count,
                )
            )

        def make_trace(
            *, success: bool, error_type: str | None = None
        ) -> RAGPipelineTrace:
            return RAGPipelineTrace(
                trace_id=current_trace_id,
                started_at=started_at,
                success=success,
                total_ms=(time.perf_counter() - started) * 1000,
                stages=tuple(stages),
                embedding_provider=self._provider_id,
                retrieved_ids=tuple(hit.id for hit in hits),
                retrieved_scores=tuple(hit.score for hit in hits),
                context_characters=len(context),
                output_characters=len(output),
                error_type=error_type,
            )

        try:
            stage_started = time.perf_counter()
            hits = await self.store.retrieve(
                question,
                k=k,
                metadata_filter=metadata_filter,
                min_score=min_score,
            )
            record_stage("retrieve", stage_started, len(hits))

            stage_started = time.perf_counter()
            context = self.store.format_context(
                hits,
                max_characters=max_context_characters,
            )
            record_stage("context", stage_started, len(context))

            stage_started = time.perf_counter()
            prompt = build_grounded_prompt(question, context)
            record_stage("prompt", stage_started, len(prompt))

            stage_started = time.perf_counter()
            output = await invoke_generator(generator, prompt)
            record_stage("generate", stage_started, len(output))

            if postprocessor is not None:
                stage_started = time.perf_counter()
                output = await self._postprocess(postprocessor, output)
                record_stage("postprocess", stage_started, len(output))

            answer = RAGAnswer(
                question=question,
                answer=output,
                context=context,
                hits=hits,
            )
            trace = make_trace(success=True)
            await self._notify(trace)
            return RAGPipelineResult(answer=answer, trace=trace)
        except Exception as exc:
            trace = make_trace(success=False, error_type=type(exc).__name__)
            await self._notify(trace)
            raise

    async def evaluate_retrieval(
        self,
        cases: Iterable[RetrievalEvaluationCase],
        *,
        k: int = 6,
        concurrency: int = 1,
    ) -> RetrievalEvaluationReport:
        """Measure hit rate, recall, MRR, and latency on reviewed relevance labels."""
        if k <= 0 or k > 1000:
            raise ValueError("k must be between 1 and 1000")
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        case_list = list(cases)
        if not case_list:
            raise ValueError("at least one evaluation case is required")
        semaphore = asyncio.Semaphore(concurrency)

        async def evaluate_case(
            case: RetrievalEvaluationCase,
        ) -> RetrievalEvaluationResult:
            async with semaphore:
                started = time.perf_counter()
                hits = await self.store.retrieve(
                    case.query,
                    k=k,
                    metadata_filter=dict(case.metadata_filter or {}),
                )
                latency_ms = (time.perf_counter() - started) * 1000
            retrieved = tuple(hit.document_id for hit in hits)
            relevant_ranks = tuple(
                rank
                for rank, document_id in enumerate(retrieved, start=1)
                if document_id in case.relevant_document_ids
            )
            retrieved_relevant = len(case.relevant_document_ids.intersection(retrieved))
            reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
            return RetrievalEvaluationResult(
                query=case.query,
                relevant_document_ids=case.relevant_document_ids,
                retrieved_document_ids=retrieved,
                relevant_ranks=relevant_ranks,
                hit=bool(relevant_ranks),
                recall_at_k=retrieved_relevant / len(case.relevant_document_ids),
                reciprocal_rank=reciprocal_rank,
                latency_ms=latency_ms,
            )

        results = tuple(
            await asyncio.gather(*(evaluate_case(case) for case in case_list))
        )
        latencies = sorted(result.latency_ms for result in results)
        p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)
        case_count = len(results)
        return RetrievalEvaluationReport(
            k=k,
            case_count=case_count,
            hit_rate_at_k=sum(result.hit for result in results) / case_count,
            mean_recall_at_k=(
                sum(result.recall_at_k for result in results) / case_count
            ),
            mean_reciprocal_rank=(
                sum(result.reciprocal_rank for result in results) / case_count
            ),
            average_latency_ms=(
                sum(result.latency_ms for result in results) / case_count
            ),
            p95_latency_ms=latencies[p95_index],
            results=results,
        )

    def _cluster_snapshot(
        self,
        metadata_filter: Mapping[str, Any] | None,
        max_vectors: int,
        max_memory_bytes: int,
    ) -> list[_ClusterRecord]:
        collection = self.store.db.get_collection(self.store.collection_name)
        requested_filter = dict(metadata_filter or {})
        records: list[_ClusterRecord] = []
        vector_bytes = 0
        for batch in collection.storage.iter_vectors():
            for vector in batch:
                metadata = dict(vector.metadata)
                if metadata.get("_rag_namespace") != self.store._namespace_token:
                    continue
                if any(
                    metadata.get(key) != value
                    for key, value in requested_filter.items()
                ):
                    continue
                document = str(metadata.get("document", ""))
                document_id = str(metadata.get("_rag_document_id", ""))
                source_value = metadata.get("source")
                source = str(source_value) if source_value is not None else None
                public_metadata = {
                    key: value
                    for key, value in metadata.items()
                    if key != "document"
                    and not key.startswith("_rag_")
                    and not key.startswith("_toucandb_")
                }
                vector_data = np.asarray(vector.data, dtype=np.float32)
                vector_bytes += int(vector_data.nbytes)
                # Snapshot arrays, the contiguous training matrix, normalization,
                # and FAISS training may briefly coexist. A conservative 4x
                # factor prevents dimensions from defeating the count guard.
                if vector_bytes * 4 > max_memory_bytes:
                    raise ValueError(
                        "clustering exceeds max_memory_bytes="
                        f"{max_memory_bytes}; filter or sample the corpus before "
                        "discovery"
                    )
                records.append(
                    _ClusterRecord(
                        id=vector.id,
                        document_id=document_id,
                        text=document,
                        metadata=public_metadata,
                        source=source,
                        vector=vector_data,
                    )
                )
                if len(records) > max_vectors:
                    raise ValueError(
                        f"clustering exceeds max_vectors={max_vectors}; filter or "
                        "sample the corpus before discovery"
                    )
        return records

    @staticmethod
    def _cluster_records(
        records: list[_ClusterRecord],
        *,
        metric: DistanceMetric,
        cluster_count: int,
        representative_count: int,
        iterations: int,
        seed: int,
    ) -> RAGClusteringReport:
        if len(records) < cluster_count * 2:
            raise ValueError(
                "semantic clustering requires at least two vectors per cluster"
            )
        matrix = np.ascontiguousarray(
            np.stack([record.vector for record in records]), dtype=np.float32
        )
        if not np.isfinite(matrix).all():
            raise ValueError("clustering vectors must contain only finite values")
        spherical = metric == DistanceMetric.COSINE
        if spherical:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)

        training_started = time.perf_counter()
        kmeans = faiss.Kmeans(
            matrix.shape[1],
            cluster_count,
            niter=iterations,
            nredo=1,
            seed=seed,
            spherical=spherical,
            min_points_per_centroid=1,
            verbose=False,
        )
        objective = float(kmeans.train(matrix))
        training_ms = (time.perf_counter() - training_started) * 1000

        assignment_started = time.perf_counter()
        raw_values, raw_assignments = kmeans.index.search(matrix, 1)
        assignments: list[list[tuple[float, int]]] = [[] for _ in range(cluster_count)]
        for record_index, (raw_value, raw_assignment) in enumerate(
            zip(raw_values[:, 0], raw_assignments[:, 0], strict=False)
        ):
            cluster_id = int(raw_assignment)
            if cluster_id < 0:
                continue
            value = float(raw_value)
            distance = max(0.0, 1.0 - value) if spherical else max(0.0, value)
            assignments[cluster_id].append((distance, record_index))
        assignment_ms = (time.perf_counter() - assignment_started) * 1000

        clusters: list[RAGCluster] = []
        for cluster_id, members in enumerate(assignments):
            if not members:
                continue
            members.sort(key=lambda item: (item[0], str(records[item[1]].id)))
            representative_hits: list[RAGHit] = []
            for distance, record_index in members[:representative_count]:
                record = records[record_index]
                score = 1.0 - distance if spherical else 1.0 / (1.0 + distance)
                representative_hits.append(
                    RAGHit(
                        id=record.id,
                        document_id=record.document_id,
                        text=record.text,
                        score=score,
                        metadata=record.metadata,
                        source=record.source,
                    )
                )
            clusters.append(
                RAGCluster(
                    cluster_id=cluster_id,
                    member_count=len(members),
                    member_ids=tuple(records[index].id for _, index in members),
                    document_ids=tuple(
                        sorted({records[index].document_id for _, index in members})
                    ),
                    representative_hits=tuple(representative_hits),
                    mean_distance=sum(distance for distance, _ in members)
                    / len(members),
                )
            )

        algorithm = "faiss-spherical-kmeans" if spherical else "faiss-kmeans"
        return RAGClusteringReport(
            algorithm=algorithm,
            vector_count=len(records),
            cluster_count=len(clusters),
            objective=objective,
            training_ms=training_ms,
            assignment_ms=assignment_ms,
            clusters=tuple(clusters),
        )

    async def discover_clusters(
        self,
        cluster_count: int,
        *,
        metadata_filter: Mapping[str, Any] | None = None,
        representative_count: int = 3,
        iterations: int = 25,
        seed: int = 1234,
        max_vectors: int = 100_000,
        max_memory_bytes: int = 512 * 1024 * 1024,
    ) -> RAGClusteringReport:
        """Discover semantic neighborhoods without persisting model-derived labels.

        The corpus snapshot and FAISS k-means work run off the event loop. The
        Explicit vector-count and working-memory guards prevent accidental
        unbounded allocations; filter or sample a larger corpus at the
        application layer.
        """
        if cluster_count <= 0:
            raise ValueError("cluster_count must be positive")
        if representative_count <= 0:
            raise ValueError("representative_count must be positive")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if max_vectors <= 0:
            raise ValueError("max_vectors must be positive")
        if max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be positive")
        records = await asyncio.to_thread(
            self._cluster_snapshot,
            metadata_filter,
            max_vectors,
            max_memory_bytes,
        )
        collection = self.store.db.get_collection(self.store.collection_name)
        return await asyncio.to_thread(
            self._cluster_records,
            records,
            metric=collection.schema.metric,
            cluster_count=cluster_count,
            representative_count=representative_count,
            iterations=iterations,
            seed=seed,
        )


__all__ = [
    "Postprocessor",
    "RAGCluster",
    "RAGClusteringReport",
    "RAGPipeline",
    "RAGPipelineObserver",
    "RAGPipelineResult",
    "RAGPipelineTrace",
    "RAGStageTrace",
    "RetrievalEvaluationCase",
    "RetrievalEvaluationReport",
    "RetrievalEvaluationResult",
]
