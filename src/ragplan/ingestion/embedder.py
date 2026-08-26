"""Pinned, local-only SentenceTransformer embedding boundary."""

from __future__ import annotations

import asyncio
import importlib
import math
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, cast

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.ingestion.chunker import HuggingFaceTokenizerAdapter, Tokenizer
from ragplan.ingestion.model_manifest import (
    EMBEDDING_DIMENSION,
    ModelArtifactManifest,
    verify_model_artifacts,
)

type EmbeddingVector = tuple[float, ...]


class Embedder(Protocol):
    """Minimal asynchronous contract consumed by ingestion and retrieval."""

    @property
    def tokenizer(self) -> Tokenizer:
        """Return the exact tokenizer paired with this embedding model."""

    async def embed_query(self, query: str) -> EmbeddingVector:
        """Embed one query exactly once."""

    async def embed_documents(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        """Embed a document/chunk batch in input order."""


class _SentenceTransformerLike(Protocol):
    tokenizer: object

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object: ...


class _SentenceTransformerFactory(Protocol):
    def __call__(
        self,
        model_name_or_path: str,
        *,
        device: str | None,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> object: ...


class _SnapshotDownload(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        revision: str,
        cache_dir: str | None,
        local_files_only: bool,
        allow_patterns: list[str],
    ) -> str: ...


def _model_incompatible(message: str) -> RAGPlanError:
    return RAGPlanError(ErrorCode.MODEL_INCOMPATIBLE, message, retryable=False)


class SentenceTransformerEmbedder:
    """384-dimensional normalized embeddings from one verified local model."""

    def __init__(
        self,
        *,
        model: _SentenceTransformerLike,
        manifest: ModelArtifactManifest,
        batch_size: int = 32,
        executor: Executor | None = None,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        model_dimension = _embedding_dimension(model)
        if (
            isinstance(model_dimension, bool)
            or model_dimension != EMBEDDING_DIMENSION
            or model_dimension != manifest.embedding_dimension
        ):
            raise _model_incompatible("loaded embedding model has an incompatible dimension")
        try:
            tokenizer = HuggingFaceTokenizerAdapter.from_tokenizer(model.tokenizer)
        except (TypeError, AttributeError) as exc:
            raise _model_incompatible("loaded embedding model has no compatible tokenizer") from exc

        self._model = model
        self._manifest = manifest
        self._batch_size = batch_size
        self._tokenizer = tokenizer
        self._executor = executor
        self._owns_executor = False

    @classmethod
    def from_local_snapshot(
        cls,
        *,
        snapshot_path: Path,
        manifest: ModelArtifactManifest,
        batch_size: int = 32,
        device: str | None = "cpu",
    ) -> SentenceTransformerEmbedder:
        """Load only a checksum-verified local snapshot, never a remote model ID."""

        verify_model_artifacts(snapshot_path, manifest)
        try:
            module = importlib.import_module("sentence_transformers")
            factory = cast(_SentenceTransformerFactory, getattr(module, "SentenceTransformer"))
            loaded = factory(
                str(snapshot_path),
                device=device,
                local_files_only=True,
                trust_remote_code=False,
            )
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise _model_incompatible("local embedding model could not be initialized") from exc
        return cls(
            model=cast(_SentenceTransformerLike, loaded),
            manifest=manifest,
            batch_size=batch_size,
        )

    @classmethod
    def from_local_cache(
        cls,
        *,
        manifest: ModelArtifactManifest,
        cache_dir: Path | None = None,
        batch_size: int = 32,
        device: str | None = "cpu",
    ) -> SentenceTransformerEmbedder:
        """Resolve the approved revision from an existing Hugging Face cache only."""

        try:
            module = importlib.import_module("huggingface_hub")
            download = cast(_SnapshotDownload, getattr(module, "snapshot_download"))
            resolved_path = download(
                repo_id=manifest.model_id,
                revision=manifest.revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=True,
                allow_patterns=sorted(manifest.artifacts),
            )
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise _model_incompatible(
                "approved embedding-model revision is not in the local cache"
            ) from exc
        return cls.from_local_snapshot(
            snapshot_path=Path(resolved_path),
            manifest=manifest,
            batch_size=batch_size,
            device=device,
        )

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def model_id(self) -> str:
        return self._manifest.model_id

    @property
    def revision(self) -> str:
        return self._manifest.revision

    @property
    def manifest_sha256(self) -> str:
        return self._manifest.sha256

    async def embed_query(self, query: str) -> EmbeddingVector:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        vectors = await self._embed((query,))
        return vectors[0]

    async def embed_documents(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        materialized = tuple(texts)
        if not all(isinstance(text, str) and text.strip() for text in materialized):
            raise ValueError("document texts must be non-empty strings")
        if not materialized:
            return ()
        return await self._embed(materialized)

    async def _embed(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        executor = self._ensure_executor()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self._encode_sync, texts)

    def _ensure_executor(self) -> Executor:
        """Return an off-loop executor so model.encode never blocks the event loop.

        Encoding one MiniLM batch takes ~10-15ms of CPU-bound work; running it
        inline stalls every concurrent request sharing the loop (deadline
        cancellations, head-of-line blocking). Callers may inject their own
        executor; otherwise a single dedicated worker is provisioned lazily.
        """

        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ragplan-embed",
            )
            self._owns_executor = True
        return self._executor

    def aclose(self) -> None:
        """Shut down a lazily provisioned executor; injected ones are untouched."""

        if self._owns_executor and self._executor is not None:
            self._executor.shutdown(wait=False)
        self._executor = None
        self._owns_executor = False

    def _encode_sync(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        try:
            encoded = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return _coerce_unit_vectors(encoded, expected_count=len(texts))
        except RAGPlanError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise _model_incompatible("embedding model returned invalid vectors") from exc


def _coerce_unit_vectors(value: object, *, expected_count: int) -> tuple[EmbeddingVector, ...]:
    rows = _materialize_iterable(value)
    if len(rows) != expected_count:
        raise _model_incompatible("embedding model returned an unexpected batch size")

    vectors: list[EmbeddingVector] = []
    for row in rows:
        components = _materialize_iterable(row)
        if len(components) != EMBEDDING_DIMENSION:
            raise _model_incompatible("embedding model returned an incompatible vector dimension")
        vector = tuple(_finite_float(component) for component in components)
        norm = math.sqrt(sum(component * component for component in vector))
        if not math.isfinite(norm) or norm <= 0.0:
            raise _model_incompatible("embedding model returned a zero or non-finite vector")
        vectors.append(tuple(component / norm for component in vector))
    return tuple(vectors)


def _materialize_iterable(value: object) -> list[object]:
    if isinstance(value, (str, bytes, Mapping)):
        raise _model_incompatible("embedding model returned an invalid vector container")
    if not isinstance(value, Iterable):
        raise _model_incompatible("embedding model returned a non-iterable vector container")
    return list(value)


def _finite_float(value: object) -> float:
    if isinstance(value, bool):
        raise _model_incompatible("embedding vector contains a non-numeric value")
    try:
        converted = float(cast(float | int, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise _model_incompatible("embedding vector contains a non-numeric value") from exc
    if not math.isfinite(converted):
        raise _model_incompatible("embedding vector contains a non-finite value")
    return converted


def _embedding_dimension(model: object) -> object:
    """Use the current Sentence Transformers API with a legacy-test fallback."""

    getter = getattr(model, "get_embedding_dimension", None)
    if not callable(getter):
        getter = getattr(model, "get_sentence_embedding_dimension", None)
    if not callable(getter):
        return None
    return getter()
