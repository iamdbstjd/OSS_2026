from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ragplan.backends.base import canonical_id_checksum
from ragplan.benchmark.contracts import CorpusChunkIndex
from ragplan.benchmark.records import BenchmarkProtocolConfig
from ragplan.core.errors import RAGPlanError
from ragplan.core.ids import canonical_chunk_id, canonical_document_id
from ragplan.core.models import Chunk
from ragplan.ingestion.frozen_benchmark import load_verified_frozen_chunks

pytestmark = pytest.mark.unit


def _fixture(tmp_path: Path) -> tuple[Path, Path, BenchmarkProtocolConfig, Chunk]:
    text = "frozen fixture evidence"
    document_id = canonical_document_id("fixture", "document-1")
    chunk = Chunk(
        id=canonical_chunk_id(document_id, 0, text),
        document_id=document_id,
        corpus_version="fixture-corpus-v1",
        position=0,
        text=text,
        token_count=3,
    )
    index = CorpusChunkIndex(
        canonical_chunk_id=chunk.id,
        document_id=document_id,
        position=0,
        token_count=3,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "chunk-index.jsonl"
    chunks_path.write_text(chunk.model_dump_json() + "\n", encoding="utf-8")
    index_path.write_text(index.model_dump_json() + "\n", encoding="utf-8")
    protocol = BenchmarkProtocolConfig.model_construct(
        corpus_version="fixture-corpus-v1",
        corpus_chunk_count=1,
        corpus_chunk_ids_sha256=canonical_id_checksum((chunk.id,)),
    )
    return chunks_path, index_path, protocol, chunk


def test_frozen_chunks_match_the_index_and_protocol(tmp_path: Path) -> None:
    chunks_path, index_path, protocol, chunk = _fixture(tmp_path)
    chunks = load_verified_frozen_chunks(
        chunks_path,
        index_path,
        protocol=protocol,
    )

    assert chunks == (chunk,)


def test_frozen_chunk_loader_rejects_a_missing_index(tmp_path: Path) -> None:
    chunks_path, _, protocol, _ = _fixture(tmp_path)
    with pytest.raises(RAGPlanError, match="chunks or index"):
        load_verified_frozen_chunks(
            chunks_path,
            tmp_path / "missing.jsonl",
            protocol=protocol,
        )
