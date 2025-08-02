"""
ToucanDB Schema Management

This module handles vector collection schemas, validation, and evolution.
It ensures type safety and data consistency across the database.
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

from .types import (
    VectorSchema, DistanceMetric, IndexType, CompressionType,
    QuantizationType, VectorId, MetadataDict
)
from .exceptions import InvalidSchemaError, ValidationError, StorageError


@dataclass
class SchemaVersion:
    """Represents a version of a schema."""
    version: int
    schema: VectorSchema
    created_at: datetime
    migration_notes: Optional[str] = None
    is_active: bool = True


class SchemaManager:
    """Manages schema creation, validation, and evolution."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.schemas_path = storage_path / "schemas"
        self.schemas_path.mkdir(parents=True, exist_ok=True)
        self._schema_cache: Dict[str, SchemaVersion] = {}
        self._load_schemas()

    def _load_schemas(self) -> None:
        """Load all schemas from disk into cache."""
        try:
            for schema_file in self.schemas_path.glob("*.json"):
                collection_name = schema_file.stem
                with open(schema_file, 'r') as f:
                    data = json.load(f)
                    schema_version = SchemaVersion(
                        version=data['version'],
                        schema=VectorSchema(**data['schema']),
                        created_at=datetime.fromisoformat(data['created_at']),
                        migration_notes=data.get('migration_notes'),
                        is_active=data.get('is_active', True)
                    )
                    self._schema_cache[collection_name] = schema_version
        except Exception as e:
            raise StorageError("load", str(self.schemas_path), f"Failed to load schemas: {e}")

    def create_schema(
        self,
        collection_name: str,
        schema: VectorSchema,
        overwrite: bool = False
    ) -> SchemaVersion:
        """Create a new schema for a collection."""
        if not overwrite and collection_name in self._schema_cache:
            raise InvalidSchemaError(
                f"Schema for collection '{collection_name}' already exists",
                {"collection_name": collection_name}
            )

        # Validate schema
        self._validate_schema(schema)

        # Create schema version
        version = 1
        if collection_name in self._schema_cache:
            version = self._schema_cache[collection_name].version + 1

        schema_version = SchemaVersion(
            version=version,
            schema=schema,
            created_at=datetime.utcnow(),
            is_active=True
        )

        # Save to disk
        self._save_schema(collection_name, schema_version)

        # Update cache
        self._schema_cache[collection_name] = schema_version

        return schema_version

    def get_schema(self, collection_name: str) -> Optional[VectorSchema]:
        """Get the current schema for a collection."""
        if collection_name in self._schema_cache:
            return self._schema_cache[collection_name].schema
        return None

    def get_schema_version(self, collection_name: str) -> Optional[SchemaVersion]:
        """Get the current schema version for a collection."""
        return self._schema_cache.get(collection_name)

    def list_collections(self) -> List[str]:
        """List all collections with schemas."""
        return [name for name, version in self._schema_cache.items() if version.is_active]

    def delete_schema(self, collection_name: str) -> bool:
        """Delete a schema (mark as inactive)."""
        if collection_name not in self._schema_cache:
            return False

        schema_version = self._schema_cache[collection_name]
        schema_version.is_active = False

        # Save updated schema
        self._save_schema(collection_name, schema_version)

        return True

    def validate_vector(
        self,
        collection_name: str,
        vector_data: List[float],
        metadata: Optional[MetadataDict] = None
    ) -> bool:
        """Validate a vector against its collection schema."""
        schema = self.get_schema(collection_name)
        if not schema:
            raise InvalidSchemaError(f"No schema found for collection '{collection_name}'")

        # Check dimensions
        if len(vector_data) != schema.dimensions:
            raise ValidationError(
                "vector_dimensions",
                len(vector_data),
                f"Expected {schema.dimensions} dimensions"
            )

        # Validate metadata if schema is defined
        if schema.metadata_schema and metadata:
            self._validate_metadata(metadata, schema.metadata_schema)

        return True

    def _validate_schema(self, schema: VectorSchema) -> None:
        """Validate a schema configuration."""
        # Check dimension limits
        if schema.dimensions <= 0:
            raise InvalidSchemaError("Dimensions must be positive")

        if schema.dimensions > 10000:
            raise InvalidSchemaError("Dimensions cannot exceed 10,000")

        # Validate index parameters
        if schema.index_type == IndexType.HNSW:
            if schema.hnsw_ef_construction < schema.hnsw_m:
                raise InvalidSchemaError(
                    "HNSW ef_construction must be >= M parameter"
                )

        if schema.index_type == IndexType.IVF:
            if schema.ivf_nlist <= 0:
                raise InvalidSchemaError("IVF nlist must be positive")

        # Validate metadata schema if provided
        if schema.metadata_schema:
            self._validate_metadata_schema(schema.metadata_schema)

    def _validate_metadata_schema(self, metadata_schema: Dict[str, str]) -> None:
        """Validate metadata schema definition."""
        supported_types = {"string", "integer", "float", "boolean", "datetime", "list", "dict"}

        for field, field_type in metadata_schema.items():
            if field_type not in supported_types:
                raise InvalidSchemaError(
                    f"Unsupported metadata type '{field_type}' for field '{field}'"
                )

    def _validate_metadata(self, metadata: MetadataDict, schema: Dict[str, str]) -> None:
        """Validate metadata against its schema."""
        for field, expected_type in schema.items():
            if field in metadata:
                value = metadata[field]
                if not self._check_type(value, expected_type):
                    raise ValidationError(
                        f"metadata.{field}",
                        value,
                        f"Expected type {expected_type}"
                    )

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected type."""
        type_map = {
            "string": str,
            "integer": int,
            "float": (int, float),
            "boolean": bool,
            "list": list,
            "dict": dict
        }

        if expected_type == "datetime":
            return isinstance(value, (str, datetime))

        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)

        return False

    def _save_schema(self, collection_name: str, schema_version: SchemaVersion) -> None:
        """Save schema to disk."""
        schema_file = self.schemas_path / f"{collection_name}.json"

        data = {
            "version": schema_version.version,
            "schema": schema_version.schema.dict(),
            "created_at": schema_version.created_at.isoformat(),
            "migration_notes": schema_version.migration_notes,
            "is_active": schema_version.is_active
        }

        try:
            with open(schema_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise StorageError("write", str(schema_file), f"Failed to save schema: {e}")

    def get_schema_hash(self, collection_name: str) -> Optional[str]:
        """Get a hash of the current schema for integrity checking."""
        schema = self.get_schema(collection_name)
        if not schema:
            return None

        schema_dict = schema.dict()
        schema_str = json.dumps(schema_dict, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()

    def migrate_schema(
        self,
        collection_name: str,
        new_schema: VectorSchema,
        migration_notes: Optional[str] = None
    ) -> SchemaVersion:
        """Migrate a schema to a new version."""
        current_version = self.get_schema_version(collection_name)
        if not current_version:
            raise InvalidSchemaError(f"No existing schema for collection '{collection_name}'")

        # Validate migration compatibility
        self._validate_migration(current_version.schema, new_schema)

        # Create new version
        new_version = SchemaVersion(
            version=current_version.version + 1,
            schema=new_schema,
            created_at=datetime.utcnow(),
            migration_notes=migration_notes,
            is_active=True
        )

        # Deactivate old version
        current_version.is_active = False

        # Save both versions
        self._save_schema(collection_name, current_version)
        self._save_schema(collection_name, new_version)

        # Update cache
        self._schema_cache[collection_name] = new_version

        return new_version

    def _validate_migration(self, old_schema: VectorSchema, new_schema: VectorSchema) -> None:
        """Validate that migration is safe and supported."""
        # Dimension changes are generally not supported
        if old_schema.dimensions != new_schema.dimensions:
            raise InvalidSchemaError(
                "Dimension changes are not supported in schema migration"
            )

        # Distance metric changes require reindexing
        if old_schema.metric != new_schema.metric:
            # This is allowed but will trigger reindexing
            pass

        # Index type changes are allowed
        if old_schema.index_type != new_schema.index_type:
            # This is allowed but will trigger reindexing
            pass

    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a collection."""
        schema_version = self.get_schema_version(collection_name)
        if not schema_version:
            return None

        return {
            "name": collection_name,
            "version": schema_version.version,
            "schema": schema_version.schema.dict(),
            "created_at": schema_version.created_at.isoformat(),
            "migration_notes": schema_version.migration_notes,
            "is_active": schema_version.is_active,
            "schema_hash": self.get_schema_hash(collection_name)
        }

    def export_schemas(self, output_path: Path) -> None:
        """Export all schemas to a single file."""
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "schemas": {}
        }

        for collection_name, schema_version in self._schema_cache.items():
            export_data["schemas"][collection_name] = {
                "version": schema_version.version,
                "schema": schema_version.schema.dict(),
                "created_at": schema_version.created_at.isoformat(),
                "migration_notes": schema_version.migration_notes,
                "is_active": schema_version.is_active
            }

        try:
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        except Exception as e:
            raise StorageError("write", str(output_path), f"Failed to export schemas: {e}")

    def import_schemas(self, input_path: Path, overwrite: bool = False) -> List[str]:
        """Import schemas from a file."""
        try:
            with open(input_path, 'r') as f:
                import_data = json.load(f)

            imported_collections = []

            for collection_name, schema_data in import_data["schemas"].items():
                if not overwrite and collection_name in self._schema_cache:
                    continue

                schema = VectorSchema(**schema_data["schema"])
                schema_version = SchemaVersion(
                    version=schema_data["version"],
                    schema=schema,
                    created_at=datetime.fromisoformat(schema_data["created_at"]),
                    migration_notes=schema_data.get("migration_notes"),
                    is_active=schema_data.get("is_active", True)
                )

                self._save_schema(collection_name, schema_version)
                self._schema_cache[collection_name] = schema_version
                imported_collections.append(collection_name)

            return imported_collections

        except Exception as e:
            raise StorageError("read", str(input_path), f"Failed to import schemas: {e}")