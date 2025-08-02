"""
ToucanDB Exception Classes

This module defines all custom exceptions used throughout ToucanDB,
providing clear error handling and debugging information.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from .types import ErrorCode


class ToucanDBException(Exception):
    """Base exception class for all ToucanDB errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class CollectionNotFoundError(ToucanDBException):
    """Raised when attempting to access a non-existent collection."""

    def __init__(self, collection_name: str):
        super().__init__(
            f"Collection '{collection_name}' not found",
            ErrorCode.COLLECTION_NOT_FOUND,
            {"collection_name": collection_name}
        )


class VectorNotFoundError(ToucanDBException):
    """Raised when attempting to access a non-existent vector."""

    def __init__(self, vector_id: str, collection_name: str):
        super().__init__(
            f"Vector '{vector_id}' not found in collection '{collection_name}'",
            ErrorCode.VECTOR_NOT_FOUND,
            {"vector_id": vector_id, "collection_name": collection_name}
        )


class DimensionMismatchError(ToucanDBException):
    """Raised when vector dimensions don't match the schema."""

    def __init__(self, expected: int, actual: int, vector_id: Optional[str] = None):
        message = f"Dimension mismatch: expected {expected}, got {actual}"
        if vector_id:
            message += f" for vector '{vector_id}'"

        super().__init__(
            message,
            ErrorCode.DIMENSION_MISMATCH,
            {"expected_dimensions": expected, "actual_dimensions": actual, "vector_id": vector_id}
        )


class InvalidSchemaError(ToucanDBException):
    """Raised when a schema is invalid or incompatible."""

    def __init__(self, reason: str, schema_details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Invalid schema: {reason}",
            ErrorCode.INVALID_SCHEMA,
            {"reason": reason, "schema": schema_details}
        )


class EncryptionError(ToucanDBException):
    """Raised when encryption/decryption operations fail."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            f"Encryption error during {operation}: {reason}",
            ErrorCode.ENCRYPTION_ERROR,
            {"operation": operation, "reason": reason}
        )


class StorageError(ToucanDBException):
    """Raised when storage operations fail."""

    def __init__(self, operation: str, path: str, reason: str):
        super().__init__(
            f"Storage error during {operation} at '{path}': {reason}",
            ErrorCode.STORAGE_ERROR,
            {"operation": operation, "path": path, "reason": reason}
        )


class IndexError(ToucanDBException):
    """Raised when index operations fail."""

    def __init__(self, operation: str, index_type: str, reason: str):
        super().__init__(
            f"Index error during {operation} with {index_type}: {reason}",
            ErrorCode.INDEX_ERROR,
            {"operation": operation, "index_type": index_type, "reason": reason}
        )


class MemoryError(ToucanDBException):
    """Raised when memory-related operations fail."""

    def __init__(self, operation: str, reason: str, memory_usage: Optional[int] = None):
        super().__init__(
            f"Memory error during {operation}: {reason}",
            ErrorCode.MEMORY_ERROR,
            {"operation": operation, "reason": reason, "memory_usage_mb": memory_usage}
        )


class PermissionDeniedError(ToucanDBException):
    """Raised when access is denied due to insufficient permissions."""

    def __init__(self, operation: str, resource: str, user_id: Optional[str] = None):
        message = f"Permission denied for {operation} on '{resource}'"
        if user_id:
            message += f" by user '{user_id}'"

        super().__init__(
            message,
            ErrorCode.PERMISSION_DENIED,
            {"operation": operation, "resource": resource, "user_id": user_id}
        )


class RateLimitExceededError(ToucanDBException):
    """Raised when rate limits are exceeded."""

    def __init__(self, operation: str, limit: int, window_seconds: int):
        super().__init__(
            f"Rate limit exceeded for {operation}: {limit} requests per {window_seconds} seconds",
            ErrorCode.RATE_LIMIT_EXCEEDED,
            {"operation": operation, "limit": limit, "window_seconds": window_seconds}
        )


class ConfigurationError(ToucanDBException):
    """Raised when configuration is invalid."""

    def __init__(self, parameter: str, value: Any, reason: str):
        super().__init__(
            f"Invalid configuration for '{parameter}' = {value}: {reason}",
            details={"parameter": parameter, "value": value, "reason": reason}
        )


class DatabaseCorruptionError(ToucanDBException):
    """Raised when database corruption is detected."""

    def __init__(self, component: str, details: str):
        super().__init__(
            f"Database corruption detected in {component}: {details}",
            ErrorCode.STORAGE_ERROR,
            {"component": component, "corruption_details": details}
        )


class ConnectionError(ToucanDBException):
    """Raised when database connection fails."""

    def __init__(self, reason: str, retry_count: int = 0):
        super().__init__(
            f"Connection failed: {reason} (retries: {retry_count})",
            details={"reason": reason, "retry_count": retry_count}
        )


class BatchOperationError(ToucanDBException):
    """Raised when batch operations fail."""

    def __init__(
        self,
        operation_type: str,
        failed_items: int,
        total_items: int,
        errors: Optional[list] = None
    ):
        super().__init__(
            f"Batch {operation_type} failed: {failed_items}/{total_items} items failed",
            details={
                "operation_type": operation_type,
                "failed_items": failed_items,
                "total_items": total_items,
                "errors": errors or []
            }
        )


class QueryTimeoutError(ToucanDBException):
    """Raised when queries exceed timeout limits."""

    def __init__(self, timeout_seconds: float, query_type: str):
        super().__init__(
            f"Query timeout after {timeout_seconds}s for {query_type}",
            details={"timeout_seconds": timeout_seconds, "query_type": query_type}
        )


class ResourceExhaustedError(ToucanDBException):
    """Raised when system resources are exhausted."""

    def __init__(self, resource_type: str, current_usage: str, limit: str):
        super().__init__(
            f"{resource_type} exhausted: {current_usage} exceeds limit of {limit}",
            details={
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit
            }
        )


class ValidationError(ToucanDBException):
    """Raised when data validation fails."""

    def __init__(self, field: str, value: Any, constraint: str):
        super().__init__(
            f"Validation failed for field '{field}': {constraint}",
            details={"field": field, "value": value, "constraint": constraint}
        )


# Utility functions for exception handling
def handle_exception(func):
    """Decorator to handle and convert common exceptions to ToucanDB exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise StorageError("read", str(e.filename or "unknown"), str(e))
        except PermissionError as e:
            raise PermissionDeniedError("file_access", str(e.filename or "unknown"))
        except MemoryError as e:
            raise MemoryError("allocation", str(e))
        except ValueError as e:
            raise ValidationError("unknown", "unknown", str(e))
        except Exception as e:
            # Re-raise ToucanDB exceptions as-is
            if isinstance(e, ToucanDBException):
                raise
            # Wrap other exceptions
            raise ToucanDBException(f"Unexpected error: {str(e)}")
    return wrapper


def create_error_context(operation: str, **kwargs) -> Dict[str, Any]:
    """Create error context for debugging."""
    return {
        "operation": operation,
        "timestamp": str(datetime.utcnow()),
        **kwargs
    }