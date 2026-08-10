"""Deterministic text preparation and token-window chunking for ingestion."""

from ragplan.ingestion.chunker import (
    ChunkerConfig,
    HuggingFaceTokenizerAdapter,
    TokenEncoding,
    Tokenizer,
    chunk_document,
)
from ragplan.ingestion.embedder import Embedder, EmbeddingVector, SentenceTransformerEmbedder
from ragplan.ingestion.model_manifest import (
    EMBEDDING_DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    ModelArtifactManifest,
    load_default_model_artifact_manifest,
    load_model_artifact_manifest,
    verify_model_artifacts,
)
from ragplan.ingestion.normalize import normalize_text

__all__ = [
    "ChunkerConfig",
    "EMBEDDING_DIMENSION",
    "Embedder",
    "EmbeddingVector",
    "HuggingFaceTokenizerAdapter",
    "MODEL_ID",
    "MODEL_REVISION",
    "ModelArtifactManifest",
    "SentenceTransformerEmbedder",
    "TokenEncoding",
    "Tokenizer",
    "chunk_document",
    "load_default_model_artifact_manifest",
    "load_model_artifact_manifest",
    "normalize_text",
    "verify_model_artifacts",
]
