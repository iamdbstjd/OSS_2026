"""Deterministic text preparation and token-window chunking for ingestion."""

from ragplan.ingestion.audit import load_graph_tier_policy
from ragplan.ingestion.chunker import (
    ChunkerConfig,
    HuggingFaceTokenizerAdapter,
    TokenEncoding,
    Tokenizer,
    chunk_document,
)
from ragplan.ingestion.embedder import Embedder, EmbeddingVector, SentenceTransformerEmbedder
from ragplan.ingestion.entities import ChunkExtraction, EntityExtractor, ParsedToken
from ragplan.ingestion.extractor_version import (
    SPACY_MODEL_NAME,
    SPACY_MODEL_VERSION,
    SPACY_VERSION,
    TOKENIZERS_VERSION,
    build_extractor_version,
)
from ragplan.ingestion.manifest import (
    ActiveCorpusPointer,
    ActiveCorpusResolver,
    ManifestRepository,
)
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
from ragplan.ingestion.pipeline import GraphExtractionResult, extract_graph
from ragplan.ingestion.reconcile import ActivationCoordinator, IngestionSource
from ragplan.ingestion.relations import extract_relations, normalize_predicate
from ragplan.ingestion.resolver import resolve_entities

__all__ = [
    "ChunkerConfig",
    "ChunkExtraction",
    "EMBEDDING_DIMENSION",
    "Embedder",
    "EmbeddingVector",
    "EntityExtractor",
    "GraphExtractionResult",
    "HuggingFaceTokenizerAdapter",
    "MODEL_ID",
    "MODEL_REVISION",
    "ManifestRepository",
    "ModelArtifactManifest",
    "ParsedToken",
    "SPACY_MODEL_NAME",
    "SPACY_MODEL_VERSION",
    "SPACY_VERSION",
    "SentenceTransformerEmbedder",
    "TokenEncoding",
    "TOKENIZERS_VERSION",
    "Tokenizer",
    "chunk_document",
    "build_extractor_version",
    "extract_graph",
    "extract_relations",
    "load_default_model_artifact_manifest",
    "load_graph_tier_policy",
    "load_model_artifact_manifest",
    "normalize_text",
    "normalize_predicate",
    "resolve_entities",
    "verify_model_artifacts",
    "ActivationCoordinator",
    "ActiveCorpusPointer",
    "ActiveCorpusResolver",
    "IngestionSource",
]
