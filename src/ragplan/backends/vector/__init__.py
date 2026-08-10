"""Vector backend interfaces and the Qdrant implementation."""

from ragplan.backends.vector.base import VectorBackend, VectorIngestionWriter
from ragplan.backends.vector.qdrant import (
    QdrantCollectionManager,
    QdrantVectorBackend,
    QdrantVectorConfig,
    QdrantVectorWriter,
    canonical_id_checksum,
    collection_name_for_version,
)

__all__ = [
    "QdrantCollectionManager",
    "QdrantVectorBackend",
    "QdrantVectorConfig",
    "QdrantVectorWriter",
    "VectorBackend",
    "VectorIngestionWriter",
    "canonical_id_checksum",
    "collection_name_for_version",
]
