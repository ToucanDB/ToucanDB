"""Correctness of the fast paths: results must match exact brute-force search.

Each test pins a behaviour that an optimisation could silently break: tombstone
exclusion inside FAISS, batched record loading, widening filtered search,
snapshot reuse, and IVF training on rebuild.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from toucandb import DatabaseConfig, SearchQuery, ToucanDB, create_schema


def _unit_rows(seed: int, count: int, dimensions: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(count, dimensions))
    matrix = matrix.astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _exact_top_ids(
    data: np.ndarray, query: np.ndarray, allowed: list[int], k: int
) -> list[int]:
    scores = data[allowed] @ query
    return [allowed[position] for position in np.argsort(-scores)[:k]]


@pytest.mark.asyncio
@pytest.mark.parametrize("index_type", ["flat", "hnsw", "ivf"])
async def test_deleted_and_replaced_vectors_never_surface(
    tmp_path: Path, index_type: str
) -> None:
    data = _unit_rows(1, 400, 16)
    db = await ToucanDB.create(tmp_path / index_type)
    await db.create_collection(create_schema("items", 16, index_type=index_type))
    inserted = await db.insert_vectors(
        "items", [{"id": i, "vector": data[i]} for i in range(400)]
    )
    assert inserted.success

    # Fewer than the compaction minimum, so these stay physical tombstones.
    deleted_ids = list(range(0, 400, 9))
    deleted = await db.delete_vectors("items", deleted_ids)
    assert deleted.success and deleted.data == len(deleted_ids)
    collection = db.get_collection("items")
    assert collection.index.tombstone_count == len(deleted_ids)

    alive = [i for i in range(400) if i not in set(deleted_ids)]
    for query_id in (0, 9, 18, 200):  # queries centred on deleted vectors
        result = await db.search_vectors(
            "items",
            SearchQuery(vector=data[query_id], k=10, include_metadata=False, nprobe=64),
        )
        assert result.success
        returned = [item["id"] for item in result.data or []]
        assert len(returned) == 10
        assert not set(returned) & set(deleted_ids)
        if index_type == "flat":
            assert returned == _exact_top_ids(data, data[query_id], alive, 10)
    await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted_count", [900, 1100])
async def test_flat_search_is_exact_on_both_sides_of_the_overfetch_threshold(
    tmp_path: Path, deleted_count: int
) -> None:
    """A flat scan over-fetches past few tombstones and uses a selector for many."""
    from toucandb.vector_engine import _FLAT_OVERFETCH_MAX_TOMBSTONES

    assert 900 <= _FLAT_OVERFETCH_MAX_TOMBSTONES < 1100
    data = _unit_rows(8, 6000, 4)
    db = await ToucanDB.create(tmp_path / "threshold")
    await db.create_collection(create_schema("items", 4, index_type="flat"))
    assert (
        await db.insert_vectors(
            "items", [{"id": i, "vector": data[i]} for i in range(6000)]
        )
    ).success
    # Under the 20% compaction ratio, so every delete remains a tombstone.
    deleted_ids = list(range(0, deleted_count * 5, 5))
    assert (await db.delete_vectors("items", deleted_ids)).success
    assert db.get_collection("items").index.tombstone_count == deleted_count

    alive = sorted(set(range(6000)) - set(deleted_ids))
    for query_id in (0, 5, 2501, 5999):
        result = await db.search_vectors(
            "items", SearchQuery(vector=data[query_id], k=25, include_metadata=False)
        )
        returned = [item["id"] for item in result.data or []]
        assert returned == _exact_top_ids(data, data[query_id], alive, 25)
    await db.close()


@pytest.mark.asyncio
async def test_tombstones_survive_snapshot_reopen(tmp_path: Path) -> None:
    data = _unit_rows(2, 200, 8)
    path = tmp_path / "reopen"
    db = await ToucanDB.create(path)
    await db.create_collection(create_schema("items", 8, index_type="flat"))
    assert (
        await db.insert_vectors(
            "items", [{"id": i, "vector": data[i]} for i in range(200)]
        )
    ).success
    # An upsert leaves the replaced physical row behind as a tombstone.
    replacement = _unit_rows(3, 1, 8)[0]
    assert (
        await db.upsert_vectors("items", [{"id": 5, "vector": replacement}])
    ).success
    assert (await db.delete_vectors("items", [7, 8])).success
    await db.close()

    db = await ToucanDB.create(path)
    collection = db.get_collection("items")
    assert collection.index.tombstone_count == 3  # restored from the snapshot
    data[5] = replacement
    alive = [i for i in range(200) if i not in {7, 8}]
    for query_id in (5, 7, 8):
        result = await db.search_vectors(
            "items", SearchQuery(vector=data[query_id], k=5, include_metadata=False)
        )
        returned = [item["id"] for item in result.data or []]
        assert returned == _exact_top_ids(data, data[query_id], alive, 5)
    await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_size_mb", [0, 64])
@pytest.mark.parametrize("index_type", ["flat", "hnsw"])
async def test_filtered_search_matches_exact_results(
    tmp_path: Path, index_type: str, cache_size_mb: int
) -> None:
    data = _unit_rows(4, 1500, 12)
    path = tmp_path / "filtered"
    config = DatabaseConfig(
        storage=DatabaseConfig.StorageConfig(path=str(path)),
        memory=DatabaseConfig.MemoryConfig(cache_size_mb=cache_size_mb),
    )
    db = await ToucanDB.create(path, config)
    await db.create_collection(create_schema("items", 12, index_type=index_type))
    assert (
        await db.insert_vectors(
            "items",
            [
                {
                    "id": i,
                    "vector": data[i],
                    "metadata": {"common": i % 3, "rare": i % 250, "all": "yes"},
                }
                for i in range(1500)
            ],
        )
    ).success

    cases = [
        ({"all": "yes"}, list(range(1500))),  # matches everything
        ({"common": 1}, [i for i in range(1500) if i % 3 == 1]),
        ({"rare": 7}, [i for i in range(1500) if i % 250 == 7]),  # only 6 exist
        ({"rare": 7, "common": 1}, [i for i in range(1500) if i % 750 == 7]),
        ({"rare": -1}, []),  # forces exhaustion of every candidate
    ]
    for metadata_filter, allowed in cases:
        result = await db.search_vectors(
            "items",
            SearchQuery(vector=data[7], k=10, metadata_filter=metadata_filter),
        )
        assert result.success, result.error_message
        items = result.data or []
        expected = _exact_top_ids(data, data[7], allowed, 10) if allowed else []
        assert len(items) == len(expected)
        for item in items:
            assert metadata_filter.items() <= item["metadata"].items()
        scores = [item["score"] for item in items]
        assert scores == sorted(scores, reverse=True)
        if index_type == "flat" or len(allowed) <= 10:
            # Exact index, or a filter so narrow that the search must widen
            # until every match is found regardless of the index.
            assert [item["id"] for item in items] == expected
    await db.close()


@pytest.mark.asyncio
async def test_batched_load_handles_mixed_ids_misses_and_cache(tmp_path: Path) -> None:
    db = await ToucanDB.create(tmp_path / "batched")
    await db.create_collection(create_schema("items", 2, index_type="flat"))
    ids: list[str | int] = [1, "1", "b", 2, "with space"]
    assert (
        await db.insert_vectors(
            "items",
            [
                {"id": vector_id, "vector": [float(n), 1.0], "metadata": {"n": n}}
                for n, vector_id in enumerate(ids)
            ],
        )
    ).success
    storage = db.get_collection("items").storage

    for cache_limit in (64 * 1024 * 1024, 0):
        storage.set_cache_limit(cache_limit)
        loaded = storage.load_vectors([*ids, "absent", 1, 999])
        assert set(loaded) == set(ids)
        # The integer 1 and the string "1" are distinct records.
        assert loaded[1].metadata == {"n": 0} and loaded["1"].metadata == {"n": 1}
        assert storage.load_vector("absent") is None
        assert storage.load_vector("b").metadata == {"n": 2}

    # More IDs than one SQL statement carries exercises the chunking.
    many = [{"id": f"m{i}", "vector": [float(i), 2.0]} for i in range(1200)]
    assert (await db.insert_vectors("items", many)).success
    assert len(storage.load_vectors([f"m{i}" for i in range(1200)])) == 1200
    await db.close()


@pytest.mark.asyncio
async def test_iteration_releases_lock_and_never_evicts_hot_records(
    tmp_path: Path,
) -> None:
    db = await ToucanDB.create(tmp_path / "iterate")
    await db.create_collection(create_schema("items", 4, index_type="flat"))
    data = _unit_rows(5, 300, 4)
    assert (
        await db.insert_vectors(
            "items", [{"id": i, "vector": data[i]} for i in range(300)]
        )
    ).success
    storage = db.get_collection("items").storage

    seen: list[int] = []
    iterator = storage.iter_vectors(batch_size=64)
    first_batch = next(iterator)
    # The storage lock is free between batches, so an abandoned or slow
    # consumer cannot block other threads.
    assert storage._lock.acquire(blocking=False)
    storage._lock.release()
    seen.extend(vector.id for vector in first_batch)
    for batch in iterator:
        seen.extend(vector.id for vector in batch)
    assert sorted(seen) == list(range(300)) and len(set(seen)) == 300

    # With room for roughly one record, a full scan must keep the hot entry.
    storage.set_cache_limit(0)
    storage.set_cache_limit(400)
    hot = storage.load_vector(42)
    assert hot is not None and 42 in storage._vector_cache
    for _ in storage.iter_vectors(batch_size=64):
        pass
    assert 42 in storage._vector_cache
    assert storage._cache_bytes <= 400
    await db.close()


@pytest.mark.asyncio
async def test_unchanged_session_does_not_rewrite_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "snapshot"
    db = await ToucanDB.create(path)
    await db.create_collection(create_schema("items", 4, index_type="flat"))
    data = _unit_rows(6, 50, 4)
    assert (
        await db.insert_vectors(
            "items", [{"id": i, "vector": data[i]} for i in range(50)]
        )
    ).success
    await db.close()

    index_dir = path / "items" / "index"
    before = sorted(entry.name for entry in index_dir.iterdir())

    db = await ToucanDB.create(path)  # read-only session
    found = await db.search_vectors("items", SearchQuery(vector=data[0], k=1))
    assert (found.data or [])[0]["id"] == 0
    await db.backup(tmp_path / "backup")
    await db.close()
    assert sorted(entry.name for entry in index_dir.iterdir()) == before
    assert (tmp_path / "backup" / "items" / "index" / "manifest.json").exists()

    db = await ToucanDB.create(path)  # a write must still refresh the snapshot
    assert (await db.insert_vectors("items", [{"id": 99, "vector": data[1]}])).success
    await db.close()
    assert sorted(entry.name for entry in index_dir.iterdir()) != before

    db = await ToucanDB.create(path)
    assert db.get_collection("items").index.active_count == 51
    await db.close()


@pytest.mark.asyncio
async def test_ivf_rebuild_trains_on_enough_examples(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    data = _unit_rows(7, 6000, 8)
    db = await ToucanDB.create(tmp_path / "ivf")
    await db.create_collection(create_schema("items", 8, index_type="ivf"))
    assert (
        await db.insert_vectors(
            "items", [{"id": i, "vector": data[i]} for i in range(6000)]
        )
    ).success
    collection = db.get_collection("items")
    capfd.readouterr()
    await collection.optimize()
    # FAISS prints this when k-means gets fewer than 39 points per centroid.
    assert "please provide at least" not in capfd.readouterr().err
    assert int(collection.index.index.nlist) == 77  # floor(sqrt(6000))
    assert collection.index.active_count == 6000

    result = await db.search_vectors(
        "items", SearchQuery(vector=data[3], k=1, include_metadata=False)
    )
    assert (result.data or [])[0]["id"] == 3
    await db.close()


@pytest.mark.asyncio
async def test_whole_operation_is_validated_before_any_batch_commits(
    tmp_path: Path,
) -> None:
    db = await ToucanDB.create(tmp_path / "atomic")
    await db.create_collection(create_schema("items", 2, index_type="flat"))
    good = [{"id": i, "vector": [float(i), 1.0]} for i in range(5)]

    for bad, code in [
        ({"id": "bad", "vector": [1.0, float("inf")]}, "VALIDATION_ERROR"),
        # Finite as a Python float, but overflows the stored float32.
        ({"id": "bad", "vector": [1.0, 1e39]}, "VALIDATION_ERROR"),
        ({"id": "bad", "vector": [1.0, "x"]}, "VALIDATION_ERROR"),
        ({"id": "bad", "vector": 3.0}, "VALIDATION_ERROR"),
        ({"id": "bad", "vector": [1.0, 2.0, 3.0]}, "DIMENSION_MISMATCH"),
        ({"id": 0, "vector": [1.0, 2.0]}, "VALIDATION_ERROR"),  # duplicate ID
        ({"id": "bad", "vector": [1.0, 2.0], "metadata": []}, "VALIDATION_ERROR"),
    ]:
        # batch_size=2 puts the bad record in a later batch than good ones.
        result = await db.insert_vectors("items", [*good, bad], batch_size=2)
        assert not result.success and result.error_code == code, bad
        assert db.list_vector_ids("items") == []

    assert (await db.insert_vectors("items", good, batch_size=2)).success
    assert sorted(db.list_vector_ids("items")) == [0, 1, 2, 3, 4]
    await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("index_type", ["flat", "hnsw", "ivf"])
async def test_exact_fallback_works_on_an_index_restored_from_a_snapshot(
    tmp_path: Path, index_type: str
) -> None:
    """A reopened index comes from ``faiss.read_index``, not a constructor.

    The exact pass reads the flat vectors behind an HNSW graph and the tombstone
    bitmap is rebuilt from the snapshot mapping, so both must survive a restart.
    """
    data = _unit_rows(9, 3000, 12)
    path = tmp_path / "restored"
    db = await ToucanDB.create(path)
    await db.create_collection(create_schema("items", 12, index_type=index_type))
    assert (
        await db.insert_vectors(
            "items",
            [
                {"id": i, "vector": data[i], "metadata": {"rare": i % 500}}
                for i in range(3000)
            ],
        )
    ).success
    # One of the six matching vectors, plus scattered non-matching ones.
    deleted_ids = [7, *range(1, 3000, 40)]
    assert (await db.delete_vectors("items", deleted_ids)).success
    await db.close()

    db = await ToucanDB.create(path)
    assert db.get_collection("items").index.tombstone_count == len(deleted_ids)
    allowed = [i for i in range(3000) if i % 500 == 7 and i != 7]
    result = await db.search_vectors(
        "items", SearchQuery(vector=data[507], k=10, metadata_filter={"rare": 7})
    )
    assert result.success, result.error_message
    returned = [item["id"] for item in result.data or []]
    assert returned == _exact_top_ids(data, data[507], allowed, 10)
    await db.close()
