from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

import pytest

from ragplan.backends.vector.qdrant import canonical_id_checksum
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import Chunk, VectorStageManifest
from ragplan.ingestion.chunker import TokenEncoding, Tokenizer
from ragplan.ingestion.embedder import EmbeddingVector
from ragplan.ingestion.service import (
    CorpusDocument,
    CorpusFile,
    load_corpus_file,
    stage_corpus,
)

pytestmark = pytest.mark.unit


class _Encoding:
    def __init__(self, text: str) -> None:
        self._tokens = text.split()

    @property
    def token_count(self) -> int:
        return len(self._tokens)

    def decode(self, start: int, end: int) -> str:
        return " ".join(self._tokens[start:end])


class _Tokenizer:
    def encode(self, text: str) -> TokenEncoding:
        return _Encoding(text)


class _Embedder:
    tokenizer: Tokenizer = _Tokenizer()

    async def embed_query(self, query: str) -> EmbeddingVector:
        raise AssertionError(query)

    async def embed_documents(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, *([0.0] * 383)) for _ in texts)


class _Writer:
    def __init__(self) -> None:
        self.chunks: tuple[Chunk, ...] = ()

    async def stage_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        corpus_version: str,
        *,
        embedding_artifact_manifest_sha256: str,
    ) -> VectorStageManifest:
        self.chunks = tuple(chunks)
        assert len(embeddings) == len(chunks)
        return VectorStageManifest(
            corpus_version=corpus_version,
            collection_name="service_fixture",
            chunk_count=len(chunks),
            canonical_id_checksum=canonical_id_checksum(tuple(item.id for item in chunks)),
            embedding_set_checksum="c" * 64,
            embedding_artifact_manifest_sha256=embedding_artifact_manifest_sha256,
        )


def test_packaged_corpus_loader_is_strict_and_accepts_sample(tmp_path: Path) -> None:
    sample = load_corpus_file(Path("examples/sample_corpus.json"))
    assert len(sample.documents) == 3

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"v1","source_dataset":"a","source_dataset":"b","documents":[]}',
        encoding="utf-8",
    )
    with pytest.raises(RAGPlanError) as captured:
        load_corpus_file(duplicate)
    assert captured.value.code is ErrorCode.INVALID_REQUEST


def test_packaged_sample_matches_the_documented_repository_example() -> None:
    packaged = files("ragplan.resources").joinpath("sample_corpus.json").read_text(encoding="utf-8")
    documented = Path("examples/sample_corpus.json").read_text(encoding="utf-8")

    assert json.loads(packaged) == json.loads(documented)


@pytest.mark.asyncio
async def test_packaged_ingest_service_uses_production_220_by_40_chunking() -> None:
    corpus = CorpusFile(
        source_dataset="fixture",
        documents=(
            CorpusDocument(
                source_document_id="doc",
                text=" ".join(f"token-{index}" for index in range(221)),
            ),
        ),
    )
    writer = _Writer()
    manifest_sha256 = hashlib.sha256(b"manifest").hexdigest()

    stage, chunks = await stage_corpus(
        corpus,
        corpus_version="fixture-v1",
        embedder=_Embedder(),
        writer=writer,
        embedding_artifact_manifest_sha256=manifest_sha256,
    )

    assert [item.token_count for item in chunks] == [220, 41]
    assert writer.chunks == chunks
    assert stage.embedding_artifact_manifest_sha256 == manifest_sha256
