"""ToucanDB's embedded vector index and durable collection storage.

SQLite is the source of truth. FAISS is a rebuildable, generation-checked
search accelerator. This split keeps writes atomic, startup fast after a clean
close, and recovery deterministic after an interrupted write or stale index.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import itertools
import json
import logging
import math
import os
import shutil
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import faiss  # type: ignore[import-not-found,import-untyped]
import lz4.frame  # type: ignore[import-not-found,import-untyped]
import msgpack  # type: ignore[import-not-found,import-untyped]
import numpy as np
from cryptography.fernet import Fernet, InvalidToken  # type: ignore[import-not-found]
from cryptography.hazmat.primitives import hashes  # type: ignore[import-not-found]
from cryptography.hazmat.primitives.kdf.pbkdf2 import (
    PBKDF2HMAC,  # type: ignore[import-not-found]
)
from cryptography.hazmat.primitives.kdf.scrypt import (
    Scrypt,  # type: ignore[import-not-found]
)

from .exceptions import EncryptionError, IndexError, StorageError, ValidationError
from .types import (
    CollectionStats,
    CompressionType,
    DistanceMetric,
    ErrorCode,
    IndexType,
    MetadataDict,
    OperationResult,
    QuantizationType,
    SearchQuery,
    SearchResult,
    Vector,
    VectorId,
    VectorSchema,
)

logger = logging.getLogger(__name__)

# FAISS clustering defaults to warning below 39 training examples per
# centroid. Exact flat search is both cheaper and at least as accurate below
# that point, so IVF collections promote only when useful training is present.
_IVF_MIN_POINTS_PER_CENTROID = 39

# A rebuild trains IVF on this many examples per centroid when the collection
# has them. It sits above the FAISS warning floor and well below the 256 at
# which FAISS starts subsampling, bounding the transient training matrix.
_IVF_TRAIN_POINTS_PER_CENTROID = 64

# SQLite's default host-parameter ceiling was 999 before 3.32.
_SQLITE_MAX_KEYS_PER_QUERY = 500

# A FAISS flat scan loses its BLAS kernel when it is given an IDSelector, which
# costs a constant ~2.4x however few IDs are excluded. Asking for
# ``k + tombstones`` keeps BLAS but grows the result heap linearly. Measured on
# 100k vectors, over-fetching wins below roughly 600 tombstones at 64 dimensions
# and 3,500 at 384, and is never more than ~30% from the better choice at this
# threshold. Graph and inverted-list scans have no BLAS path to lose, so they
# always use the selector.
_FLAT_OVERFETCH_MAX_TOMBSTONES = 1024


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Write a file durably and expose it with one atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(payload)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class IndexConfig:
    """Configuration for FAISS indices."""

    ef_construction: int = 200
    ef_search: int = 100
    m: int = 16
    nlist: int = 1024
    nprobe: int = 64
    metric: DistanceMetric = DistanceMetric.COSINE


class VectorIndex:
    """In-memory FAISS index with logical-ID tombstone tracking."""

    def __init__(self, schema: VectorSchema, config: IndexConfig):
        self.schema = schema
        self.config = config
        self.index: Any | None = None
        self.id_mapping: dict[int, VectorId] = {}
        self.reverse_mapping: dict[VectorId, int] = {}
        self.next_id = 0
        # One bit per physical FAISS ID, set while the ID is active. FAISS reads
        # it through an IDSelector so tombstones are skipped inside the search
        # instead of being over-fetched and discarded in Python.
        self._active_bitmap: np.ndarray[Any, Any] = np.zeros(0, dtype=np.uint8)
        self._lock = threading.RLock()
        self._initialize_index()

    @property
    def _faiss_metric(self) -> int:
        if self.schema.metric in {
            DistanceMetric.COSINE,
            DistanceMetric.DOT_PRODUCT,
        }:
            return cast(int, faiss.METRIC_INNER_PRODUCT)
        if self.schema.metric == DistanceMetric.EUCLIDEAN:
            return cast(int, faiss.METRIC_L2)
        raise IndexError(
            "initialize",
            self.schema.index_type.value,
            f"Unsupported distance metric: {self.schema.metric.value}",
        )

    @property
    def physical_count(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    @property
    def active_count(self) -> int:
        return len(self.reverse_mapping)

    @property
    def tombstone_count(self) -> int:
        return max(0, self.physical_count - self.active_count)

    def _initialize_index(self, training_size: int = 0) -> None:
        """Initialize an index; IVF is sized lazily from available training data."""
        dimensions = self.schema.dimensions
        metric = self._faiss_metric

        if self.schema.index_type == IndexType.FLAT:
            self.index = (
                faiss.IndexFlatIP(dimensions)
                if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(dimensions)
            )
        elif self.schema.index_type == IndexType.HNSW:
            self.index = faiss.IndexHNSWFlat(
                dimensions,
                self.config.m,
                metric,
            )
            self.index.hnsw.efConstruction = self.config.ef_construction
        elif self.schema.index_type == IndexType.IVF:
            if training_size <= 0:
                self.index = None
                return
            if training_size < _IVF_MIN_POINTS_PER_CENTROID:
                self.index = (
                    faiss.IndexFlatIP(dimensions)
                    if metric == faiss.METRIC_INNER_PRODUCT
                    else faiss.IndexFlatL2(dimensions)
                )
                return
            # Training an IVF index with more centroids than examples fails.
            # sqrt(n) is a conservative starting point, while the points-per-
            # centroid bound avoids under-trained partitions and FAISS warning
            # churn. Neither may exceed the explicit schema cap.
            nlist = min(
                self.config.nlist,
                max(1, int(math.sqrt(training_size))),
                max(1, training_size // _IVF_MIN_POINTS_PER_CENTROID),
            )
            quantizer = (
                faiss.IndexFlatIP(dimensions)
                if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(dimensions)
            )
            self.index = faiss.IndexIVFFlat(
                quantizer,
                dimensions,
                nlist,
                metric,
            )
        else:  # pragma: no cover - rejected by VectorSchema
            raise IndexError(
                "initialize",
                self.schema.index_type.value,
                f"Unsupported index type: {self.schema.index_type.value}",
            )

    @property
    def training_target(self) -> int:
        """Examples the next ``add_vectors`` call should carry to train well."""
        with self._lock:
            if self.index is None or self.index.is_trained:
                return 0
            return int(self.index.nlist) * _IVF_TRAIN_POINTS_PER_CENTROID

    def _mark_active(self, start: int, stop: int) -> None:
        required = (stop + 7) >> 3
        if required > len(self._active_bitmap):
            grown = np.zeros(max(required, 2 * len(self._active_bitmap)), np.uint8)
            grown[: len(self._active_bitmap)] = self._active_bitmap
            self._active_bitmap = grown
        ids = np.arange(start, stop, dtype=np.int64)
        np.bitwise_or.at(
            self._active_bitmap,
            ids >> 3,
            np.left_shift(1, ids & 7).astype(np.uint8),
        )

    def _mark_inactive(self, faiss_id: int) -> None:
        byte = faiss_id >> 3
        self._active_bitmap[byte] = int(self._active_bitmap[byte]) & ~(
            1 << (faiss_id & 7)
        )

    def reset(self, expected_size: int = 0) -> None:
        """Reset both FAISS and logical-ID mappings."""
        with self._lock:
            self.id_mapping.clear()
            self.reverse_mapping.clear()
            self.next_id = 0
            self._active_bitmap = np.zeros(0, dtype=np.uint8)
            self._initialize_index(expected_size)

    def restore(self, index: Any, mapping: dict[int, VectorId], next_id: int) -> None:
        """Adopt a validated snapshot and derive its reverse map and bitmap."""
        with self._lock:
            self.index = index
            self.id_mapping = mapping
            self.reverse_mapping = {value: key for key, value in mapping.items()}
            self.next_id = next_id
            active = np.zeros(next_id, dtype=bool)
            if mapping:
                active[np.fromiter(mapping, dtype=np.int64, count=len(mapping))] = True
            self._active_bitmap = np.packbits(active, bitorder="little")

    def add_vectors(self, vectors: list[Vector]) -> list[int]:
        """Add vectors and return their positional FAISS IDs."""
        with self._lock:
            if not vectors:
                return []

            vector_data = np.stack(
                [np.asarray(vector.data, dtype=np.float32) for vector in vectors]
            )
            if not np.isfinite(vector_data).all():
                raise ValidationError(
                    "vector", "non-finite", "Vectors must contain only finite values"
                )

            if self.schema.metric == DistanceMetric.COSINE:
                norms = np.linalg.norm(vector_data, axis=1, keepdims=True)
                vector_data = vector_data / np.maximum(norms, 1e-12)

            if self.index is None:
                self._initialize_index(len(vectors))
            if self.index is None:  # pragma: no cover - defensive
                raise IndexError("add", self.schema.index_type.value, "Missing index")

            if not self.index.is_trained:
                self.index.train(vector_data)

            start_id = self.next_id
            faiss_ids = list(range(start_id, start_id + len(vectors)))
            self.index.add(vector_data)
            for faiss_id, vector in zip(faiss_ids, vectors, strict=False):
                self.id_mapping[faiss_id] = vector.id
                self.reverse_mapping[vector.id] = faiss_id
            self.next_id += len(vectors)
            self._mark_active(start_id, self.next_id)
            return faiss_ids

    def _search_parameters(
        self,
        query: SearchQuery,
        active_limit: int,
        exact: bool,
        exclude_tombstones: bool,
    ) -> Any | None:
        """Per-query FAISS parameters; never mutates state shared by searches."""
        options: dict[str, Any] = {}
        if exclude_tombstones:
            # The selector borrows the bitmap's memory, which stays valid because
            # every mutation and search runs under this index's lock. Its first
            # argument is the bitmap length in bytes, not the number of IDs.
            options["sel"] = faiss.IDSelectorBitmap(
                len(self._active_bitmap),
                faiss.swig_ptr(self._active_bitmap),
            )
        if self.schema.index_type == IndexType.HNSW and not exact:
            return faiss.SearchParametersHNSW(
                efSearch=query.ef or self.config.ef_search,
                **options,
            )
        # An IVF schema below its training floor is still a flat index.
        nlist = getattr(self.index, "nlist", None)
        if self.schema.index_type == IndexType.IVF and nlist is not None:
            requested_nprobe = query.nprobe or self.config.nprobe
            if exact or (query.metadata_filter and active_limit >= self.active_count):
                # Probing every list makes IVF an exact search.
                requested_nprobe = int(nlist)
            return faiss.SearchParametersIVF(
                nprobe=min(int(nlist), requested_nprobe),
                **options,
            )
        return faiss.SearchParameters(**options) if options else None

    def search(self, query: SearchQuery, active_limit: int) -> list[SearchResult]:
        """Return the closest active candidates; tombstones never surface."""
        with self._lock:
            return list(self.search_candidates(query, active_limit)[0])

    def search_candidates(
        self,
        query: SearchQuery,
        active_limit: int,
        *,
        exact: bool = False,
        skip: Collection[VectorId] = (),
    ) -> tuple[Iterator[SearchResult], bool]:
        """Return ranked candidates and whether the index came up short.

        An approximate index can supply fewer neighbours than requested even
        when that many are active: an HNSW graph is not fully reachable, and
        IVF only sees the lists it probes. The flag lets a caller that must
        consider every vector retry with ``exact=True``, which scans the flat
        vectors HNSW keeps alongside its graph or probes every IVF list.

        Candidates are built lazily, best first, because a filtered search
        usually stops long before the end of a wide candidate list; IDs in
        ``skip`` are never built at all. Consume the iterator before the index
        is mutated, as ``VectorCollection`` does under its operation lock.
        """
        with self._lock:
            if self.index is None or self.active_count == 0 or active_limit <= 0:
                return iter(()), False

            query_vector = np.asarray(query.vector, dtype=np.float32).reshape(1, -1)
            if not np.isfinite(query_vector).all():
                raise ValidationError(
                    "query.vector",
                    "non-finite",
                    "Query vector must contain only finite values",
                )
            if self.schema.metric == DistanceMetric.COSINE:
                norm = float(np.linalg.norm(query_vector))
                if norm > 0:
                    query_vector = query_vector / norm

            searched = self.index
            if exact and self.schema.index_type == IndexType.HNSW:
                searched = faiss.downcast_index(self.index.storage)
            is_flat_scan = not hasattr(searched, "hnsw") and not hasattr(
                searched, "nlist"
            )
            tombstones = self.tombstone_count
            overfetch = is_flat_scan and tombstones <= _FLAT_OVERFETCH_MAX_TOMBSTONES
            requested = min(self.active_count, active_limit)
            distances, indices = searched.search(
                query_vector,
                # Tombstones are dropped below; ``requested`` never exceeds the
                # active count, so this stays within the physical count.
                requested + tombstones if overfetch else requested,
                params=self._search_parameters(
                    query,
                    active_limit,
                    exact,
                    exclude_tombstones=tombstones > 0 and not overfetch,
                ),
            )
            # FAISS pads with -1, always at the end, when it runs out.
            found = int(np.count_nonzero(indices[0] != -1))
            came_up_short = found < len(indices[0])
            faiss_ids: list[int] = indices[0][:found].tolist()
            raw_distances: list[float] = distances[0][:found].tolist()

        similarity = self.schema.metric in {
            DistanceMetric.COSINE,
            DistanceMetric.DOT_PRODUCT,
        }

        def ranked() -> Iterator[SearchResult]:
            produced = 0
            for faiss_id, distance in zip(faiss_ids, raw_distances, strict=True):
                vector_id = self.id_mapping.get(faiss_id)
                if vector_id is None:
                    continue  # a tombstone that an over-fetching flat scan saw
                score = distance if similarity else 1.0 / (1.0 + distance)
                if query.threshold is not None and score < query.threshold:
                    # FAISS returns best-first, so every later score is lower.
                    return
                produced += 1
                if vector_id not in skip:
                    yield SearchResult(
                        id=vector_id,
                        vector=None,
                        score=score,
                        metadata={},
                        distance=distance,
                    )
                if produced >= active_limit:
                    return

        return ranked(), came_up_short

    def remove_vector(self, vector_id: VectorId) -> bool:
        """Logically remove a vector; HNSW is compacted by collection policy."""
        with self._lock:
            faiss_id = self.reverse_mapping.pop(vector_id, None)
            if faiss_id is None:
                return False
            self.id_mapping.pop(faiss_id, None)
            self._mark_inactive(faiss_id)
            return True

    def get_stats(self) -> dict[str, Any]:
        """Return logical and physical index statistics."""
        return {
            "total_vectors": self.active_count,
            "physical_vectors": self.physical_count,
            "tombstones": self.tombstone_count,
            "index_type": self.schema.index_type,
            "metric": self.schema.metric,
            "dimensions": self.schema.dimensions,
            "estimated_vector_bytes": self.physical_count * self.schema.dimensions * 4,
            "is_trained": bool(self.index is not None and self.index.is_trained),
        }


class CompressionEngine:
    """Vector quantization and payload compression."""

    @staticmethod
    def compress_data(data: bytes, compression_type: CompressionType) -> bytes:
        if compression_type == CompressionType.NONE:
            return data
        if compression_type == CompressionType.LZ4:
            return cast(bytes, lz4.frame.compress(data))
        raise ValidationError(
            "compression_type", compression_type, "Unsupported compression type"
        )

    @staticmethod
    def decompress_data(data: bytes, compression_type: CompressionType) -> bytes:
        if compression_type == CompressionType.NONE:
            return data
        if compression_type == CompressionType.LZ4:
            return cast(bytes, lz4.frame.decompress(data))
        raise ValidationError(
            "compression_type", compression_type, "Unsupported compression type"
        )

    @staticmethod
    def quantize_vector(
        vector: np.ndarray[Any, Any], quantization: QuantizationType
    ) -> np.ndarray[Any, Any]:
        if quantization == QuantizationType.NONE:
            return vector.astype(np.float32)
        if quantization == QuantizationType.FP16:
            return vector.astype(np.float16)
        if quantization == QuantizationType.INT8:
            maximum = float(np.max(np.abs(vector))) if vector.size else 0.0
            scale = maximum / 127.0 if maximum > 0 else 1.0
            quantized: np.ndarray[Any, Any] = np.clip(
                np.rint(vector / scale), -127, 127
            ).astype(np.int8)
            return quantized
        raise ValidationError(
            "quantization", quantization, "Unsupported quantization type"
        )

    @classmethod
    def quantize_for_storage(
        cls,
        vector: np.ndarray[Any, Any],
        quantization: QuantizationType,
    ) -> tuple[np.ndarray[Any, Any], float | None]:
        if quantization != QuantizationType.INT8:
            return cls.quantize_vector(vector, quantization), None
        maximum = float(np.max(np.abs(vector))) if vector.size else 0.0
        scale = maximum / 127.0 if maximum > 0 else 1.0
        quantized = np.clip(np.rint(vector / scale), -127, 127).astype(np.int8)
        return quantized, scale


class EncryptionEngine:
    """Fernet encryption with a per-collection salt and legacy read support."""

    LEGACY_SALT = b"toucandb_salt_2023"
    LEGACY_ITERATIONS = 100_000

    def __init__(self, key: str | None, salt_path: Path):
        self.fernet: Fernet | None = None
        self.legacy_fernet: Fernet | None = None
        self._password: bytes | None = None
        if key is None:
            return

        try:
            if salt_path.exists():
                salt = salt_path.read_bytes()
                if len(salt) != 16:
                    raise ValueError("Encryption salt must be exactly 16 bytes")
            else:
                salt = os.urandom(16)
                _atomic_write(salt_path, salt)
            password = key.encode("utf-8")
            self._password = password
            kdf = Scrypt(
                salt=salt,
                length=32,
                n=2**14,
                r=8,
                p=1,
            )
            self.fernet = Fernet(base64.urlsafe_b64encode(kdf.derive(password)))
        except Exception as exc:
            raise EncryptionError("initialize", str(exc)) from exc

    @staticmethod
    def _derive(password: bytes, salt: bytes, iterations: int) -> Fernet:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        return Fernet(base64.urlsafe_b64encode(kdf.derive(password)))

    @property
    def enabled(self) -> bool:
        return self.fernet is not None

    def encrypt(self, data: bytes) -> bytes:
        if self.fernet is None:
            return data
        try:
            return cast(bytes, self.fernet.encrypt(data))
        except Exception as exc:
            raise EncryptionError("encrypt", str(exc)) from exc

    def decrypt(self, data: bytes) -> bytes:
        if self.fernet is None:
            return data
        try:
            return cast(bytes, self.fernet.decrypt(data))
        except InvalidToken as exc:
            raise EncryptionError(
                "decrypt", "Invalid encryption key or payload"
            ) from exc
        except Exception as exc:
            raise EncryptionError("decrypt", str(exc)) from exc

    def legacy_candidates(self, data: bytes) -> Iterator[bytes]:
        """Yield encrypted and plaintext interpretations of a legacy payload."""
        if self.legacy_fernet is None and self._password is not None:
            self.legacy_fernet = self._derive(
                self._password,
                self.LEGACY_SALT,
                self.LEGACY_ITERATIONS,
            )
        if self.legacy_fernet is not None:
            try:
                yield cast(bytes, self.legacy_fernet.decrypt(data))
            except InvalidToken:
                pass
        yield data


class VectorStorage:
    """Atomic SQLite storage with a byte-bounded LRU read cache."""

    FORMAT_VERSION = 2
    KEY_CHECK = b"ToucanDB encryption check v1"

    def __init__(
        self,
        storage_path: Path,
        schema: VectorSchema,
        encryption_key: str | None = None,
        *,
        cache_size_bytes: int = 64 * 1024 * 1024,
        encryption_engine: EncryptionEngine | None = None,
    ):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.database_path = storage_path / "vectors.sqlite3"
        self.compression = CompressionEngine()
        self.encryption = encryption_engine or EncryptionEngine(
            encryption_key, storage_path / "encryption.salt"
        )
        self._lock = threading.RLock()
        self._cache_limit = max(0, cache_size_bytes)
        self._cache_bytes = 0
        self._vector_cache: OrderedDict[VectorId, tuple[Vector, int]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self.database_path,
                timeout=5.0,
                check_same_thread=False,
            )
            self._connection = connection
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA temp_store=MEMORY")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._initialize_database()
            self._validate_legacy_key_before_initializing_encryption()
            self._verify_encryption_mode()
            self._migrate_legacy_storage()
        except Exception as exc:
            if connection is not None:
                connection.close()
            if isinstance(exc, (EncryptionError, StorageError)):
                raise
            raise StorageError("open", str(self.database_path), str(exc)) from exc

    def _initialize_database(self) -> None:
        with self._connection:
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    storage_key TEXT PRIMARY KEY,
                    vector_id BLOB NOT NULL,
                    payload BLOB NOT NULL,
                    raw_size INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                ) WITHOUT ROWID
                """)
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL
                ) WITHOUT ROWID
                """)
            self._connection.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES('generation', '0')"
            )

    def _get_setting(self, key: str) -> bytes | None:
        row = self._connection.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value = row[0]
        return value.encode("utf-8") if isinstance(value, str) else cast(bytes, value)

    def _set_setting(self, key: str, value: bytes) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO settings(key, value) VALUES(?, ?)",
            (key, value),
        )

    def _verify_encryption_mode(self) -> None:
        with self._lock, self._connection:
            stored = self._get_setting("encryption_check")
            if stored is None:
                prefix = b"fernet\0" if self.encryption.enabled else b"plain\0"
                token = self.encryption.encrypt(self.KEY_CHECK)
                self._set_setting("encryption_check", prefix + token)
                return

            if stored.startswith(b"fernet\0"):
                if not self.encryption.enabled:
                    raise EncryptionError(
                        "open", "This collection requires its encryption key"
                    )
                if self.encryption.decrypt(stored[7:]) != self.KEY_CHECK:
                    raise EncryptionError("open", "Invalid encryption key")
            elif stored.startswith(b"plain\0"):
                if self.encryption.enabled:
                    raise EncryptionError(
                        "open",
                        "Cannot enable encryption on an existing plaintext collection; "
                        "export and re-import it into an encrypted collection",
                    )
                if stored[6:] != self.KEY_CHECK:
                    raise EncryptionError("open", "Invalid plaintext key marker")
            else:
                raise EncryptionError("open", "Unknown encryption marker format")

    @staticmethod
    def _validate_id(vector_id: VectorId) -> None:
        if isinstance(vector_id, bool) or not isinstance(vector_id, (str, int)):
            raise ValidationError(
                "id", vector_id, "Vector IDs must be non-boolean strings or integers"
            )
        if isinstance(vector_id, str) and not vector_id:
            raise ValidationError("id", vector_id, "Vector IDs cannot be empty")

    @classmethod
    def _encode_id(cls, vector_id: VectorId) -> bytes:
        cls._validate_id(vector_id)
        kind = "string" if isinstance(vector_id, str) else "integer"
        return cast(
            bytes,
            msgpack.packb({"type": kind, "value": vector_id}, use_bin_type=True),
        )

    @staticmethod
    def _decode_id(payload: bytes) -> VectorId:
        decoded = msgpack.unpackb(payload, raw=False)
        value = decoded["value"]
        if decoded["type"] == "string" and isinstance(value, str):
            return value
        if decoded["type"] == "integer" and isinstance(value, int):
            return value
        raise ValueError("Stored vector ID has an invalid type")

    @classmethod
    def _storage_key(cls, vector_id: VectorId) -> str:
        return hashlib.sha256(cls._encode_id(vector_id)).hexdigest()

    # Fixed per-entry allowance for the Vector object, its ID, and timestamp.
    _CACHE_ENTRY_OVERHEAD = 256

    def _serialize_vector(self, vector: Vector) -> tuple[bytes, bytes, int, int]:
        """Return the encoded ID, stored payload, raw size, and cache weight."""
        self._validate_id(vector.id)
        data = np.asarray(vector.data, dtype=np.float32)
        if data.ndim != 1 or len(data) != self.schema.dimensions:
            raise ValidationError(
                "vector",
                data.shape,
                f"Expected a one-dimensional {self.schema.dimensions}-value vector",
            )
        if not np.isfinite(data).all():
            raise ValidationError(
                "vector", "non-finite", "Vectors must contain only finite values"
            )

        quantized, quantization_scale = self.compression.quantize_for_storage(
            data,
            self.schema.quantization,
        )
        stored_data = quantized.tobytes()
        record = {
            "format": self.FORMAT_VERSION,
            "id": vector.id,
            "data": stored_data,
            "shape": quantized.shape,
            "dtype": str(quantized.dtype),
            "quantization_scale": quantization_scale,
            "metadata": vector.metadata,
            "timestamp": vector.timestamp.isoformat(),
        }
        try:
            raw = cast(bytes, msgpack.packb(record, use_bin_type=True))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "metadata",
                vector.metadata,
                f"Metadata must be MessagePack-serializable: {exc}",
            ) from exc
        compressed = self.compression.compress_data(raw, self.schema.compression)
        encrypted = self.encryption.encrypt(compressed)
        weight = self._cache_weight(data.nbytes, len(raw), len(stored_data))
        return self._encode_id(vector.id), encrypted, len(raw), weight

    def _deserialize_vector(self, payload: bytes) -> tuple[Vector, int]:
        """Decode one stored payload into a vector and its cache weight."""
        decrypted = self.encryption.decrypt(payload)
        raw = self.compression.decompress_data(
            decrypted,
            self.schema.compression,
        )
        record = msgpack.unpackb(raw, raw=False)
        array = np.frombuffer(
            record["data"],
            dtype=np.dtype(record["dtype"]),
        ).reshape(tuple(record["shape"]))
        scale = record.get("quantization_scale")
        if scale is not None and array.dtype == np.int8:
            vector_data = array.astype(np.float32) * float(scale)
        else:
            vector_data = array.astype(np.float32)
        vector = Vector(
            id=record["id"],
            data=vector_data,
            metadata=record.get("metadata", {}),
            timestamp=datetime.fromisoformat(record["timestamp"]),
        )
        weight = self._cache_weight(vector_data.nbytes, len(raw), len(record["data"]))
        return vector, weight

    @classmethod
    def _cache_weight(cls, data_bytes: int, raw_size: int, stored_data: int) -> int:
        """Estimate resident size from sizes serialization already produced.

        The raw record minus its vector bytes is the packed metadata plus a few
        fixed fields, so no second MessagePack pass over the metadata is needed.
        """
        return data_bytes + max(0, raw_size - stored_data) + cls._CACHE_ENTRY_OVERHEAD

    def _cache_get(self, vector_id: VectorId) -> Vector | None:
        cached = self._vector_cache.get(vector_id)
        if cached is None:
            self._cache_misses += 1
            return None
        self._vector_cache.move_to_end(vector_id)
        self._cache_hits += 1
        return cached[0]

    def _cache_put(self, vector: Vector, weight: int, *, evict: bool = True) -> None:
        """Cache a record; with ``evict=False`` only free room is ever used."""
        if self._cache_limit <= 0:
            return
        if not evict and (
            vector.id in self._vector_cache
            or self._cache_bytes + weight > self._cache_limit
        ):
            return
        previous = self._vector_cache.pop(vector.id, None)
        if previous is not None:
            self._cache_bytes -= previous[1]
        if weight > self._cache_limit:
            return
        self._vector_cache[vector.id] = (vector, weight)
        self._cache_bytes += weight
        while self._cache_bytes > self._cache_limit and self._vector_cache:
            _, (_, evicted_weight) = self._vector_cache.popitem(last=False)
            self._cache_bytes -= evicted_weight

    def set_cache_limit(self, cache_size_bytes: int) -> None:
        """Adjust this collection's cache budget and evict excess entries."""
        with self._lock:
            self._cache_limit = max(0, cache_size_bytes)
            while self._cache_bytes > self._cache_limit and self._vector_cache:
                _, (_, evicted_weight) = self._vector_cache.popitem(last=False)
                self._cache_bytes -= evicted_weight

    @property
    def generation(self) -> int:
        with self._lock:
            raw = self._get_setting("generation") or b"0"
            return int(raw.decode("ascii"))

    def _increment_generation(self) -> int:
        generation = self.generation + 1
        self._set_setting("generation", str(generation).encode("ascii"))
        return generation

    def store_vectors(self, vectors: list[Vector]) -> None:
        """Atomically insert or replace a batch of vectors."""
        if not vectors:
            return
        prepared = []
        weights = []
        for vector in vectors:
            vector_id, payload, raw_size, weight = self._serialize_vector(vector)
            weights.append(weight)
            prepared.append(
                (
                    # Same digest as _storage_key, without encoding the ID twice.
                    hashlib.sha256(vector_id).hexdigest(),
                    vector_id,
                    payload,
                    raw_size,
                    vector.timestamp.isoformat(),
                )
            )
        try:
            with self._lock:
                with self._connection:
                    self._connection.executemany(
                        """
                        INSERT INTO vectors(
                            storage_key, vector_id, payload, raw_size, updated_at
                        ) VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(storage_key) DO UPDATE SET
                            vector_id = excluded.vector_id,
                            payload = excluded.payload,
                            raw_size = excluded.raw_size,
                            updated_at = excluded.updated_at
                        """,
                        prepared,
                    )
                    self._increment_generation()
                # Only a committed record may be served from the cache.
                for vector, weight in zip(vectors, weights, strict=True):
                    self._cache_put(vector, weight)
        except (ValidationError, EncryptionError):
            raise
        except Exception as exc:
            raise StorageError("write", str(self.database_path), str(exc)) from exc

    def store_vector(self, vector: Vector) -> None:
        """Compatibility wrapper for single-vector storage."""
        self.store_vectors([vector])

    def load_vector(self, vector_id: VectorId) -> Vector | None:
        return self.load_vectors([vector_id]).get(vector_id)

    def load_vectors(self, vector_ids: Sequence[VectorId]) -> dict[VectorId, Vector]:
        """Load many vectors, reading every cache miss with one query per chunk.

        Missing IDs are simply absent from the returned mapping.
        """
        for vector_id in vector_ids:
            self._validate_id(vector_id)
        found: dict[VectorId, Vector] = {}
        with self._lock:
            missing: dict[str, VectorId] = {}
            for vector_id in vector_ids:
                if vector_id in found:
                    continue
                cached = self._cache_get(vector_id)
                if cached is not None:
                    found[vector_id] = cached
                else:
                    missing[self._storage_key(vector_id)] = vector_id
            try:
                keys = list(missing)
                for start in range(0, len(keys), _SQLITE_MAX_KEYS_PER_QUERY):
                    chunk = keys[start : start + _SQLITE_MAX_KEYS_PER_QUERY]
                    placeholders = ",".join("?" * len(chunk))
                    rows = self._connection.execute(
                        "SELECT storage_key, payload FROM vectors "
                        f"WHERE storage_key IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    for storage_key, payload in rows:
                        vector, weight = self._deserialize_vector(cast(bytes, payload))
                        if vector.id != missing[storage_key]:
                            raise ValueError(
                                "Vector ID hash collision or corrupt record"
                            )
                        self._cache_put(vector, weight)
                        found[vector.id] = vector
            except (EncryptionError, ValidationError):
                raise
            except Exception as exc:
                raise StorageError("read", str(self.database_path), str(exc)) from exc
        return found

    def iter_vectors(self, batch_size: int = 2048) -> Iterator[list[Vector]]:
        """Stream all vectors in bounded batches, ordered by storage key.

        Each batch is one keyset-paginated query, so neither the storage lock
        nor a SQLite cursor is held while the consumer works or if it abandons
        the iterator. A scan only fills free cache room; evicting hot records
        in favour of a one-off sequential read would make later searches slower.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        last_key = ""
        while True:
            with self._lock:
                try:
                    rows = self._connection.execute(
                        "SELECT storage_key, payload FROM vectors "
                        "WHERE storage_key > ? ORDER BY storage_key LIMIT ?",
                        (last_key, batch_size),
                    ).fetchall()
                    vectors = []
                    for _, payload in rows:
                        vector, weight = self._deserialize_vector(cast(bytes, payload))
                        self._cache_put(vector, weight, evict=False)
                        vectors.append(vector)
                except (EncryptionError, ValidationError):
                    raise
                except Exception as exc:
                    raise StorageError(
                        "read", str(self.database_path), str(exc)
                    ) from exc
            if not rows:
                return
            last_key = cast(str, rows[-1][0])
            yield vectors

    def delete_vector(self, vector_id: VectorId) -> bool:
        return self.delete_vectors([vector_id]) > 0

    def delete_vectors(self, vector_ids: Sequence[VectorId]) -> int:
        """Delete a batch in one transaction and return the number removed."""
        if not vector_ids:
            return 0
        if len(set(vector_ids)) != len(vector_ids):
            raise ValidationError(
                "ids", vector_ids, "Vector IDs must be unique within a delete batch"
            )
        for vector_id in vector_ids:
            self._validate_id(vector_id)
        storage_keys = [(self._storage_key(vector_id),) for vector_id in vector_ids]
        try:
            with self._lock:
                with self._connection:
                    # For executemany, rowcount is the total across all keys.
                    cursor = self._connection.executemany(
                        "DELETE FROM vectors WHERE storage_key = ?",
                        storage_keys,
                    )
                    deleted = max(0, cursor.rowcount)
                    if deleted > 0:
                        self._increment_generation()
                if deleted > 0:
                    for vector_id in vector_ids:
                        cached = self._vector_cache.pop(vector_id, None)
                        if cached is not None:
                            self._cache_bytes -= cached[1]
                return deleted
        except (ValidationError, EncryptionError):
            raise
        except Exception as exc:
            raise StorageError("delete", str(self.database_path), str(exc)) from exc

    def list_vectors(self) -> list[VectorId]:
        with self._lock:
            try:
                rows = self._connection.execute(
                    "SELECT vector_id FROM vectors ORDER BY storage_key"
                ).fetchall()
                return [self._decode_id(cast(bytes, row[0])) for row in rows]
            except Exception as exc:
                raise StorageError("list", str(self.database_path), str(exc)) from exc

    def count(self) -> int:
        """Number of stored vectors, without the payload scan of full stats."""
        with self._lock:
            try:
                row = self._connection.execute("SELECT COUNT(*) FROM vectors")
                return int(row.fetchone()[0])
            except Exception as exc:
                raise StorageError("count", str(self.database_path), str(exc)) from exc

    def get_storage_stats(self) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0), "
                "COALESCE(SUM(raw_size), 0) FROM vectors"
            ).fetchone()
            vector_count, stored_size, raw_size = (int(value) for value in row)
            physical_size = sum(
                path.stat().st_size
                for path in (
                    self.database_path,
                    Path(f"{self.database_path}-wal"),
                    Path(f"{self.database_path}-shm"),
                )
                if path.exists()
            )
            total_reads = self._cache_hits + self._cache_misses
            return {
                "total_vectors": vector_count,
                "total_size_bytes": physical_size,
                "stored_payload_bytes": stored_size,
                "cache_entries": len(self._vector_cache),
                "cache_bytes": self._cache_bytes,
                "cache_hit_ratio": (
                    self._cache_hits / total_reads if total_reads else 0.0
                ),
                "compression_ratio": (stored_size / raw_size if raw_size else 1.0),
                "compression": self.schema.compression,
                "quantization": self.schema.quantization,
                "generation": self.generation,
            }

    def backup_to(self, destination: Path) -> None:
        """Create a transactionally consistent SQLite backup."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._lock:
                backup = sqlite3.connect(destination)
                try:
                    self._connection.backup(backup)
                finally:
                    backup.close()
        except Exception as exc:
            raise StorageError("backup", str(destination), str(exc)) from exc

    def close(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._connection.close()
            self._vector_cache.clear()
            self._cache_bytes = 0

    def _decode_legacy_payload(self, payload: bytes) -> dict[str, Any]:
        last_error: Exception | None = None
        for candidate in self.encryption.legacy_candidates(payload):
            try:
                decompressed = self.compression.decompress_data(
                    candidate,
                    self.schema.compression,
                )
                return cast(dict[str, Any], msgpack.unpackb(decompressed, raw=False))
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Cannot decode legacy payload: {last_error}")

    def _validate_legacy_key_before_initializing_encryption(self) -> None:
        """Avoid committing a new key marker before a legacy key is validated."""
        if self._get_setting("encryption_check") is not None:
            return
        vectors_path = self.storage_path / "vectors"
        first_vector = next(iter(sorted(vectors_path.glob("*.vec"))), None)
        if first_vector is None:
            return
        metadata_file = self.storage_path / "metadata" / f"{first_vector.stem}.meta"
        if not metadata_file.exists():
            raise StorageError(
                "migrate",
                str(metadata_file),
                "Legacy vector is missing its metadata pair",
            )
        try:
            self._decode_legacy_payload(first_vector.read_bytes())
            self._decode_legacy_payload(metadata_file.read_bytes())
        except Exception as exc:
            raise EncryptionError(
                "migrate",
                "Cannot decode legacy records; verify the 1.x encryption key",
            ) from exc

    def _migrate_legacy_storage(self) -> None:
        """Import the 1.0/1.1 file-per-vector format exactly once."""
        with self._lock:
            if self._get_setting("legacy_migration_complete") == b"1":
                return
        vectors_path = self.storage_path / "vectors"
        metadata_path = self.storage_path / "metadata"
        legacy_files = sorted(vectors_path.glob("*.vec"))
        if not legacy_files:
            with self._lock, self._connection:
                self._set_setting("legacy_migration_complete", b"1")
            return

        migrated: list[Vector] = []
        try:
            for vector_file in legacy_files:
                metadata_file = metadata_path / f"{vector_file.stem}.meta"
                if not metadata_file.exists():
                    raise ValueError(f"Missing metadata file for {vector_file.name}")
                vector_record = self._decode_legacy_payload(vector_file.read_bytes())
                metadata_record = self._decode_legacy_payload(
                    metadata_file.read_bytes()
                )
                array = np.frombuffer(
                    vector_record["data"],
                    dtype=np.dtype(vector_record["dtype"]),
                ).reshape(tuple(vector_record["shape"]))
                migrated.append(
                    Vector(
                        id=vector_record["id"],
                        data=array.astype(np.float32),
                        metadata=metadata_record.get("metadata", {}),
                        timestamp=datetime.fromisoformat(vector_record["timestamp"]),
                    )
                )
                if len(migrated) >= 1000:
                    self.store_vectors(migrated)
                    migrated.clear()
            self.store_vectors(migrated)
            with self._lock, self._connection:
                self._set_setting("legacy_migration_complete", b"1")
            logger.info("Migrated legacy vector files in %s", self.storage_path)
        except Exception as exc:
            raise StorageError(
                "migrate", str(self.storage_path), f"Legacy migration failed: {exc}"
            ) from exc


class VectorCollection:
    """A durable vector collection and its rebuildable FAISS accelerator."""

    SNAPSHOT_INTERVAL = 10_000
    COMPACTION_MIN_TOMBSTONES = 64
    COMPACTION_RATIO = 0.20

    def __init__(
        self,
        name: str,
        schema: VectorSchema,
        storage_path: Path,
        encryption_key: str | None = None,
        *,
        cache_size_bytes: int = 64 * 1024 * 1024,
        encryption_engine: EncryptionEngine | None = None,
        index_snapshot_interval: int = SNAPSHOT_INTERVAL,
        compaction_min_tombstones: int = COMPACTION_MIN_TOMBSTONES,
        compaction_ratio: float = COMPACTION_RATIO,
    ):
        self.name = name
        self.schema = schema
        self.storage_path = storage_path / name
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._operation_lock = threading.RLock()
        self._closed = False
        self._mutations_since_snapshot = 0
        # Storage generation that the on-disk snapshot mirrors, or None when the
        # in-memory index has diverged from it (or no snapshot exists yet).
        self._snapshot_generation: int | None = None
        self._snapshot_interval = index_snapshot_interval
        self._compaction_min_tombstones = compaction_min_tombstones
        self._compaction_ratio = compaction_ratio

        index_config = IndexConfig(
            ef_construction=schema.hnsw_ef_construction,
            m=schema.hnsw_m,
            nlist=schema.ivf_nlist,
            metric=schema.metric,
        )
        self.index = VectorIndex(schema, index_config)
        self.storage = VectorStorage(
            self.storage_path,
            schema,
            encryption_key,
            cache_size_bytes=cache_size_bytes,
            encryption_engine=encryption_engine,
        )
        now = _utcnow()
        self._stats = CollectionStats(
            name=name,
            total_vectors=0,
            dimensions=schema.dimensions,
            size_bytes=0,
            index_type=schema.index_type,
            metric=schema.metric,
            created_at=now,
            updated_at=now,
            avg_search_latency_ms=0.0,
            cache_hit_ratio=0.0,
            compression_ratio=1.0,
        )
        try:
            self._load_index_or_rebuild()
        except Exception:
            self.storage.close()
            raise

    @property
    def _snapshot_path(self) -> Path:
        return self.storage_path / "index"

    @property
    def _schema_hash(self) -> str:
        encoded = json.dumps(
            self.schema.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_snapshot(self) -> bool:
        manifest_path = self._snapshot_path / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            generation = int(manifest["generation"])
            if generation != self.storage.generation:
                return False
            if manifest["schema_hash"] != self._schema_hash:
                return False
            index_path = self._snapshot_path / manifest["index_file"]
            mapping_path = self._snapshot_path / manifest["mapping_file"]
            loaded_index = faiss.read_index(str(index_path))
            mapping_record = msgpack.unpackb(
                mapping_path.read_bytes(),
                raw=False,
                strict_map_key=False,
            )
            mapping = {int(key): value for key, value in mapping_record["ids"].items()}
            next_id = int(mapping_record["next_id"])
            if int(loaded_index.ntotal) != next_id:
                return False
            if any(key < 0 or key >= next_id for key in mapping):
                return False
            if len(set(mapping.values())) != len(mapping):
                return False
            self.index.restore(loaded_index, mapping, next_id)
            self._snapshot_generation = generation
            return True
        except Exception as exc:
            logger.warning("Ignoring invalid index snapshot for %s: %s", self.name, exc)
            return False

    def _save_snapshot(self) -> None:
        if self.index.index is None:
            return
        generation = self.storage.generation
        snapshot_path = self._snapshot_path
        snapshot_path.mkdir(parents=True, exist_ok=True)
        token = f"{generation}-{uuid.uuid4().hex}"
        index_name = f"index-{token}.faiss"
        mapping_name = f"mapping-{token}.msgpack"
        index_path = snapshot_path / index_name
        mapping_path = snapshot_path / mapping_name
        temporary_index = snapshot_path / f".{index_name}.tmp"
        try:
            faiss.write_index(self.index.index, str(temporary_index))
            os.replace(temporary_index, index_path)
            mapping_payload = cast(
                bytes,
                msgpack.packb(
                    {
                        "ids": self.index.id_mapping,
                        "next_id": self.index.next_id,
                    },
                    use_bin_type=True,
                ),
            )
            _atomic_write(mapping_path, mapping_payload)
            manifest = json.dumps(
                {
                    "format": 1,
                    "generation": generation,
                    "schema_hash": self._schema_hash,
                    "index_file": index_name,
                    "mapping_file": mapping_name,
                },
                sort_keys=True,
            ).encode("utf-8")
            _atomic_write(snapshot_path / "manifest.json", manifest)
            for candidate in snapshot_path.glob("index-*.faiss"):
                if candidate.name != index_name:
                    candidate.unlink(missing_ok=True)
            for candidate in snapshot_path.glob("mapping-*.msgpack"):
                if candidate.name != mapping_name:
                    candidate.unlink(missing_ok=True)
            self._mutations_since_snapshot = 0
            self._snapshot_generation = generation
        except Exception as exc:
            raise StorageError("snapshot", str(snapshot_path), str(exc)) from exc
        finally:
            temporary_index.unlink(missing_ok=True)

    def _save_snapshot_if_stale(self) -> None:
        """Skip rewriting an index file that already mirrors this generation.

        Serializing FAISS costs time proportional to the whole index, which a
        read-only session or a repeated backup should not pay again.
        """
        if self._snapshot_generation != self.storage.generation:
            self._save_snapshot()

    def _load_index_or_rebuild(self) -> None:
        with self._operation_lock:
            if not self._load_snapshot():
                self._rebuild_index(save_snapshot=True)
            self._stats.total_vectors = self.index.active_count

    def _rebuild_index(self, *, save_snapshot: bool) -> None:
        total = self.storage.count()
        # The rebuilt layout no longer matches any snapshot on disk.
        self._snapshot_generation = None
        self.index.reset(total)
        # Rows arrive ordered by a SHA-256 storage key, so the leading rows are
        # a uniform sample. Hold them back until IVF has enough to train every
        # centroid instead of training on whatever the first batch contains.
        training_target = self.index.training_target
        pending: list[Vector] = []
        for vectors in self.storage.iter_vectors():
            if len(pending) + len(vectors) < training_target:
                pending.extend(vectors)
                continue
            self.index.add_vectors(pending + vectors if pending else vectors)
            pending = []
            training_target = 0
        self.index.add_vectors(pending)
        self._stats.total_vectors = self.index.active_count
        if save_snapshot and total > 0:
            self._save_snapshot()

    def _maybe_compact_or_snapshot(self) -> None:
        physical = self.index.physical_count
        tombstones = self.index.tombstone_count
        if self.schema.index_type == IndexType.IVF and self.index.index is not None:
            active_count = self.index.active_count
            if not hasattr(self.index.index, "nlist"):
                if active_count >= _IVF_MIN_POINTS_PER_CENTROID:
                    self._rebuild_index(save_snapshot=True)
                    return
            else:
                current_nlist = int(self.index.index.nlist)
                target_nlist = min(
                    self.index.config.nlist,
                    max(1, int(math.sqrt(max(1, active_count)))),
                    max(1, active_count // _IVF_MIN_POINTS_PER_CENTROID),
                )
                if target_nlist >= current_nlist * 2:
                    self._rebuild_index(save_snapshot=True)
                    return
        if (
            tombstones >= self._compaction_min_tombstones
            and physical > 0
            and tombstones / physical >= self._compaction_ratio
        ):
            self._rebuild_index(save_snapshot=True)
        elif self._mutations_since_snapshot >= self._snapshot_interval:
            self._save_snapshot()

    def _validate_metadata(self, metadata: MetadataDict) -> None:
        metadata_schema = self.schema.metadata_schema
        if not metadata_schema:
            return
        type_map: dict[str, type[Any] | tuple[type[Any], ...]] = {
            "string": str,
            "integer": int,
            "float": (int, float),
            "boolean": bool,
            "datetime": (str, datetime),
            "list": list,
            "dict": dict,
        }
        for field, expected_name in metadata_schema.items():
            if field not in metadata:
                continue
            value = metadata[field]
            if expected_name == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif expected_name == "float":
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            else:
                valid = isinstance(value, type_map[expected_name])
            if not valid:
                raise ValidationError(
                    f"metadata.{field}", value, f"Expected type {expected_name}"
                )

    def validate_metadata(self, metadata: Any) -> None:
        """Validate metadata before a multi-batch operation commits anything."""
        if not isinstance(metadata, dict):
            raise ValidationError("metadata", metadata, "Metadata must be a dictionary")
        self._validate_metadata(metadata)
        try:
            msgpack.packb(metadata, use_bin_type=True)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "metadata",
                metadata,
                f"Metadata must be MessagePack-serializable: {exc}",
            ) from exc

    def prepare_vectors(
        self,
        vectors: Sequence[dict[str, Any]],
    ) -> OperationResult[list[Vector]]:
        """Validate raw insert records once and convert them to vectors.

        Nothing is written, so a caller can vet a whole multi-batch operation
        before its first batch commits and then insert the prepared vectors
        without paying for the same checks a second time.
        """
        start_time = time.perf_counter()
        try:
            vector_objects: list[Vector] = []
            seen_ids: set[VectorId] = set()
            timestamp = _utcnow()
            for vector_data in vectors:
                if "id" not in vector_data or "vector" not in vector_data:
                    raise ValidationError(
                        "vectors",
                        vector_data,
                        "Each vector needs 'id' and 'vector' fields",
                    )
                vector_id = cast(VectorId, vector_data["id"])
                VectorStorage._validate_id(vector_id)
                if vector_id in seen_ids:
                    raise ValidationError(
                        "id", vector_id, "Duplicate vector ID in batch"
                    )
                seen_ids.add(vector_id)
                try:
                    # A value too large for float32 becomes inf and is rejected
                    # by the finite check below, so the cast warning is noise.
                    with np.errstate(over="ignore"):
                        raw_vector = np.asarray(vector_data["vector"], dtype=np.float32)
                except (TypeError, ValueError) as exc:
                    raise ValidationError(
                        "vector",
                        "non-numeric",
                        "Vectors must be one-dimensional finite numeric sequences",
                    ) from exc
                if raw_vector.ndim == 0:
                    raise ValidationError(
                        "vector",
                        "scalar",
                        "Vector values must be a one-dimensional sequence",
                    )
                if raw_vector.ndim != 1 or len(raw_vector) != self.schema.dimensions:
                    actual = (
                        raw_vector.shape if raw_vector.ndim != 1 else len(raw_vector)
                    )
                    return OperationResult.error_result(
                        ErrorCode.DIMENSION_MISMATCH,
                        f"Expected {self.schema.dimensions} dimensions, "
                        f"got {actual}",
                        (time.perf_counter() - start_time) * 1000,
                    )
                if not np.isfinite(raw_vector).all():
                    raise ValidationError(
                        "vector",
                        "non-finite",
                        "Vectors must contain only finite values",
                    )
                metadata = vector_data.get("metadata", {})
                self.validate_metadata(metadata)
                vector_objects.append(
                    Vector(
                        id=vector_id,
                        data=raw_vector,
                        metadata=dict(metadata),
                        timestamp=timestamp,
                    )
                )
            return OperationResult.success_result(
                vector_objects,
                (time.perf_counter() - start_time) * 1000,
            )
        except ValidationError as exc:
            return OperationResult.error_result(
                ErrorCode.VALIDATION_ERROR,
                str(exc),
                (time.perf_counter() - start_time) * 1000,
            )
        except Exception as exc:
            # A record that is not even dictionary-shaped is invalid input too.
            return OperationResult.error_result(
                ErrorCode.VALIDATION_ERROR,
                f"Malformed vector record: {exc}",
                (time.perf_counter() - start_time) * 1000,
            )

    def _insert_sync(
        self,
        vectors: list[dict[str, Any]],
        *,
        upsert: bool,
    ) -> OperationResult[list[VectorId]]:
        prepared = self.prepare_vectors(vectors)
        if not prepared.success or prepared.data is None:
            return OperationResult.error_result(
                prepared.error_code or ErrorCode.VALIDATION_ERROR,
                prepared.error_message or "Invalid vectors",
                prepared.execution_time_ms,
            )
        result = self._insert_prepared_sync(prepared.data, upsert=upsert)
        result.execution_time_ms += prepared.execution_time_ms
        return result

    def _insert_prepared_sync(
        self,
        vector_objects: list[Vector],
        *,
        upsert: bool,
    ) -> OperationResult[list[VectorId]]:
        start_time = time.perf_counter()
        try:
            with self._operation_lock:
                existing_ids = {
                    vector.id
                    for vector in vector_objects
                    if vector.id in self.index.reverse_mapping
                }
                if existing_ids and not upsert:
                    duplicate = sorted(str(value) for value in existing_ids)[0]
                    return OperationResult.error_result(
                        ErrorCode.VALIDATION_ERROR,
                        f"Vector ID already exists: {duplicate!r}; use upsert=True",
                        (time.perf_counter() - start_time) * 1000,
                    )

                new_count = (
                    self.index.active_count + len(vector_objects) - len(existing_ids)
                )
                if (
                    self.schema.max_vectors is not None
                    and new_count > self.schema.max_vectors
                ):
                    return OperationResult.error_result(
                        ErrorCode.VALIDATION_ERROR,
                        f"Collection limit of {self.schema.max_vectors} "
                        "vectors exceeded",
                        (time.perf_counter() - start_time) * 1000,
                    )

                unchanged_ids: set[VectorId] = set()
                if upsert and existing_ids:
                    persisted_vectors = self.storage.load_vectors(list(existing_ids))
                    for vector in vector_objects:
                        persisted = persisted_vectors.get(vector.id)
                        if (
                            persisted is not None
                            and persisted.metadata == vector.metadata
                            and np.array_equal(persisted.data, vector.data)
                        ):
                            unchanged_ids.add(vector.id)

                changed_vectors = [
                    vector
                    for vector in vector_objects
                    if vector.id not in unchanged_ids
                ]
                replaced_ids = existing_ids - unchanged_ids
                self.storage.store_vectors(changed_vectors)
                try:
                    for vector_id in replaced_ids:
                        self.index.remove_vector(vector_id)
                    self.index.add_vectors(changed_vectors)
                except Exception:
                    # SQLite has already committed and remains authoritative.
                    # Rebuild immediately so the current process is consistent.
                    self._rebuild_index(save_snapshot=False)

                self._stats.total_vectors = self.index.active_count
                if changed_vectors:
                    self._stats.updated_at = _utcnow()
                    self._mutations_since_snapshot += len(changed_vectors)
                    self._maybe_compact_or_snapshot()
                return OperationResult.success_result(
                    [vector.id for vector in vector_objects],
                    (time.perf_counter() - start_time) * 1000,
                )
        except ValidationError as exc:
            return OperationResult.error_result(
                ErrorCode.VALIDATION_ERROR,
                str(exc),
                (time.perf_counter() - start_time) * 1000,
            )
        except Exception as exc:
            return OperationResult.error_result(
                ErrorCode.STORAGE_ERROR,
                str(exc),
                (time.perf_counter() - start_time) * 1000,
            )

    async def insert(
        self,
        vectors: list[dict[str, Any]],
        *,
        upsert: bool = False,
    ) -> OperationResult[list[VectorId]]:
        """Insert or upsert without blocking the caller's event loop."""
        return await asyncio.to_thread(self._insert_sync, vectors, upsert=upsert)

    async def _insert_prepared(
        self,
        vectors: list[Vector],
        *,
        upsert: bool = False,
    ) -> OperationResult[list[VectorId]]:
        """Insert one ``prepare_vectors`` result, or a contiguous slice of it.

        Private because it trusts that result's checks, including ID uniqueness.
        """
        return await asyncio.to_thread(
            self._insert_prepared_sync, vectors, upsert=upsert
        )

    @staticmethod
    def _matches_filter(metadata: MetadataDict, filter_dict: dict[str, Any]) -> bool:
        return all(metadata.get(key) == value for key, value in filter_dict.items())

    def _collect_matches(
        self,
        query: SearchQuery,
        candidates: Iterator[SearchResult],
        rejected: set[VectorId],
        matched: dict[VectorId, Vector],
    ) -> list[SearchResult]:
        """Attach stored fields to ranked candidates and apply the filter.

        Records are read in chunks, each one query, rather than row by row. The
        first chunk assumes every candidate matches, so an unfiltered search or
        a filter that matches everything loads exactly ``k`` records; later
        chunks grow with the observed rejection rate.
        """
        needs_load = (
            query.include_metadata
            or query.include_vectors
            or query.metadata_filter is not None
        )
        if not needs_load:
            return list(itertools.islice(candidates, query.k))

        results: list[SearchResult] = []
        while len(results) < query.k:
            remaining = query.k - len(results)
            judged = len(rejected) + len(matched)
            if query.metadata_filter and judged:
                match_rate = max(len(matched), 1) / judged
                chunk_size = max(remaining, math.ceil(remaining / match_rate))
            else:
                chunk_size = remaining
            chunk_size = min(chunk_size, _SQLITE_MAX_KEYS_PER_QUERY)
            chunk = list(itertools.islice(candidates, chunk_size))
            if not chunk:
                break

            unknown = [
                result.id
                for result in chunk
                if result.id not in rejected and result.id not in matched
            ]
            loaded = self.storage.load_vectors(unknown) if unknown else {}
            for result in chunk:
                if result.id in rejected:
                    continue
                vector = matched.get(result.id) or loaded.get(result.id)
                if vector is None:
                    raise StorageError(
                        "search",
                        str(self.storage.database_path),
                        f"Index points to missing vector {result.id!r}",
                    )
                if query.metadata_filter:
                    if not self._matches_filter(vector.metadata, query.metadata_filter):
                        rejected.add(result.id)
                        continue
                    matched[result.id] = vector
                if query.include_metadata or query.metadata_filter:
                    result.metadata = vector.metadata
                if query.include_vectors:
                    result.vector = vector.data
                results.append(result)
                if len(results) >= query.k:
                    break
        return results

    def _search_sync(self, query: SearchQuery) -> OperationResult[list[SearchResult]]:
        start_time = time.perf_counter()
        try:
            if len(query.vector) != self.schema.dimensions:
                return OperationResult.error_result(
                    ErrorCode.DIMENSION_MISMATCH,
                    f"Query vector has {len(query.vector)} dimensions, "
                    f"expected {self.schema.dimensions}",
                    (time.perf_counter() - start_time) * 1000,
                )
            with self._operation_lock:
                active_count = self.index.active_count
                initial_limit = (
                    query.k if not query.metadata_filter else max(32, query.k * 4)
                )
                candidate_limit = min(active_count, initial_limit)
                filtered_results: list[SearchResult] = []
                # Filter verdicts survive widening rounds, so a wider FAISS
                # search never reloads a record that was already judged.
                rejected: set[VectorId] = set()
                matched: dict[VectorId, Vector] = {}
                exact = False
                while candidate_limit > 0:
                    candidates, came_up_short = self.index.search_candidates(
                        query, candidate_limit, exact=exact, skip=rejected
                    )
                    filtered_results = self._collect_matches(
                        query, candidates, rejected, matched
                    )
                    if (
                        len(filtered_results) >= query.k
                        or not query.metadata_filter
                        or exact
                    ):
                        break
                    if came_up_short or candidate_limit >= active_count:
                        # Widening an approximate search further cannot reach
                        # the remaining vectors, so a filter that is still
                        # unsatisfied gets one exact pass over all of them.
                        exact = True
                        candidate_limit = active_count
                    else:
                        candidate_limit = min(active_count, candidate_limit * 2)

                execution_time = (time.perf_counter() - start_time) * 1000
                self._stats.avg_search_latency_ms = (
                    execution_time
                    if self._stats.avg_search_latency_ms == 0
                    else self._stats.avg_search_latency_ms * 0.9 + execution_time * 0.1
                )
                return OperationResult.success_result(
                    filtered_results[: query.k], execution_time
                )
        except ValidationError as exc:
            return OperationResult.error_result(
                ErrorCode.VALIDATION_ERROR,
                str(exc),
                (time.perf_counter() - start_time) * 1000,
            )
        except Exception as exc:
            return OperationResult.error_result(
                ErrorCode.INDEX_ERROR,
                str(exc),
                (time.perf_counter() - start_time) * 1000,
            )

    async def search(self, query: SearchQuery) -> OperationResult[list[SearchResult]]:
        """Search without blocking the caller's event loop."""
        return await asyncio.to_thread(self._search_sync, query)

    def _delete_sync(self, vector_id: VectorId) -> OperationResult[bool]:
        start_time = time.perf_counter()
        try:
            with self._operation_lock:
                storage_deleted = self.storage.delete_vector(vector_id)
                index_deleted = self.index.remove_vector(vector_id)
                deleted = storage_deleted or index_deleted
                if storage_deleted != index_deleted:
                    self._rebuild_index(save_snapshot=False)
                if deleted:
                    self._stats.total_vectors = self.index.active_count
                    self._stats.updated_at = _utcnow()
                    self._mutations_since_snapshot += 1
                    self._maybe_compact_or_snapshot()
                return OperationResult.success_result(
                    deleted,
                    (time.perf_counter() - start_time) * 1000,
                )
        except ValidationError as exc:
            return OperationResult.error_result(
                ErrorCode.VALIDATION_ERROR,
                str(exc),
                (time.perf_counter() - start_time) * 1000,
            )
        except Exception as exc:
            return OperationResult.error_result(
                ErrorCode.STORAGE_ERROR,
                str(exc),
                (time.perf_counter() - start_time) * 1000,
            )

    async def delete(self, vector_id: VectorId) -> OperationResult[bool]:
        return await asyncio.to_thread(self._delete_sync, vector_id)

    def _delete_many_sync(self, vector_ids: Sequence[VectorId]) -> OperationResult[int]:
        start_time = time.perf_counter()
        try:
            with self._operation_lock:
                storage_deleted = self.storage.delete_vectors(vector_ids)
                index_deleted = sum(
                    self.index.remove_vector(vector_id) for vector_id in vector_ids
                )
                if storage_deleted != index_deleted:
                    self._rebuild_index(save_snapshot=False)
                if storage_deleted:
                    self._stats.total_vectors = self.index.active_count
                    self._stats.updated_at = _utcnow()
                    self._mutations_since_snapshot += storage_deleted
                    self._maybe_compact_or_snapshot()
                return OperationResult.success_result(
                    storage_deleted,
                    (time.perf_counter() - start_time) * 1000,
                )
        except ValidationError as exc:
            return OperationResult.error_result(
                ErrorCode.VALIDATION_ERROR,
                str(exc),
                (time.perf_counter() - start_time) * 1000,
            )
        except Exception as exc:
            return OperationResult.error_result(
                ErrorCode.STORAGE_ERROR,
                str(exc),
                (time.perf_counter() - start_time) * 1000,
            )

    async def delete_many(self, vector_ids: Sequence[VectorId]) -> OperationResult[int]:
        """Delete many vectors with one durable transaction."""
        return await asyncio.to_thread(self._delete_many_sync, vector_ids)

    async def optimize(self) -> None:
        """Compact tombstones, retrain IVF, and persist a fresh index snapshot."""
        await asyncio.to_thread(self._optimize_sync)

    def _optimize_sync(self) -> None:
        with self._operation_lock:
            self._rebuild_index(save_snapshot=True)

    def get_stats(self) -> CollectionStats:
        storage_stats = self.storage.get_storage_stats()
        self._stats.total_vectors = self.index.active_count
        index_size = (
            sum(
                path.stat().st_size
                for path in self._snapshot_path.rglob("*")
                if path.is_file()
            )
            if self._snapshot_path.exists()
            else 0
        )
        self._stats.size_bytes = storage_stats["total_size_bytes"] + index_size
        self._stats.cache_hit_ratio = storage_stats["cache_hit_ratio"]
        self._stats.compression_ratio = storage_stats["compression_ratio"]
        return self._stats

    def backup_to(self, destination: Path) -> None:
        with self._operation_lock:
            self._save_snapshot_if_stale()
            destination.mkdir(parents=True, exist_ok=True)
            self.storage.backup_to(destination / "vectors.sqlite3")
            salt = self.storage_path / "encryption.salt"
            if salt.exists():
                shutil.copy2(salt, destination / salt.name)
            if self._snapshot_path.exists():
                shutil.copytree(
                    self._snapshot_path,
                    destination / "index",
                    dirs_exist_ok=True,
                )

    def close(self) -> None:
        with self._operation_lock:
            if self._closed:
                return
            self._save_snapshot_if_stale()
            self.storage.close()
            self._closed = True
