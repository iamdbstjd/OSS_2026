from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from ragplan.backends.vector.qdrant import canonical_id_checksum
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import Chunk, ChunkerVersion, VectorStageManifest
from ragplan.ingestion.chunker import TokenEncoding, Tokenizer
from ragplan.ingestion.embedder import EmbeddingVector

pytestmark = pytest.mark.unit


def _load_command() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "ingest.py"
    spec = importlib.util.spec_from_file_location("ragplan_script_ingest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


command = _load_command()


class _WhitespaceEncoding:
    def __init__(self, text: str) -> None:
        self.tokens = text.split()

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def decode(self, start: int, end: int) -> str:
        return " ".join(self.tokens[start:end])


class _WhitespaceTokenizer:
    def encode(self, text: str) -> TokenEncoding:
        return _WhitespaceEncoding(text)


class _FakeEmbedder:
    def __init__(self) -> None:
        self.tokenizer: Tokenizer = _WhitespaceTokenizer()
        self.document_calls: list[tuple[str, ...]] = []

    async def embed_query(self, query: str) -> EmbeddingVector:
        raise AssertionError(f"query embedding is not expected: {len(query)}")

    async def embed_documents(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        materialized = tuple(texts)
        self.document_calls.append(materialized)
        return tuple((1.0, *([0.0] * 383)) for _ in materialized)


class _FakeWriter:
    def __init__(self) -> None:
        self.chunks: tuple[Chunk, ...] = ()
        self.embeddings: tuple[Sequence[float], ...] = ()

    async def stage_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        corpus_version: str,
        *,
        embedding_artifact_manifest_sha256: str,
        chunker_version: ChunkerVersion,
    ) -> VectorStageManifest:
        self.chunks = tuple(chunks)
        self.embeddings = tuple(embeddings)
        return VectorStageManifest(
            corpus_version=corpus_version,
            collection_name="sample_collection",
            chunk_count=len(chunks),
            canonical_id_checksum=canonical_id_checksum(
                tuple(chunk.canonical_chunk_id for chunk in chunks)
            ),
            embedding_set_checksum="c" * 64,
            embedding_artifact_manifest_sha256=embedding_artifact_manifest_sha256,
            chunker_version=chunker_version,
        )


def _write_corpus(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_strict_corpus_loader_accepts_sample_and_rejects_extra_fields(tmp_path: Path) -> None:
    sample = command.load_corpus_file(Path("examples/sample_corpus.json"))
    assert sample.schema_version == "v1"
    assert sample.source_dataset == "ragplan-stage3-sample"
    assert len(sample.documents) == 3

    invalid = tmp_path / "invalid.json"
    _write_corpus(
        invalid,
        {
            "schema_version": "v1",
            "source_dataset": "dataset",
            "documents": [{"source_document_id": "doc", "text": "content", "extra": 1}],
        },
    )
    with pytest.raises(RAGPlanError) as captured:
        command.load_corpus_file(invalid)
    assert captured.value.code is ErrorCode.INVALID_REQUEST


def test_strict_corpus_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"v1","source_dataset":"one","source_dataset":"two","documents":[]}',
        encoding="utf-8",
    )

    with pytest.raises(RAGPlanError, match="strict JSON"):
        command.load_corpus_file(path)


@pytest.mark.asyncio
async def test_ingest_uses_220_by_40_chunker_one_embed_batch_and_stage_writer() -> None:
    corpus = command.CorpusFile(
        schema_version="v1",
        source_dataset="unit",
        documents=(
            command.CorpusDocument(
                source_document_id="doc",
                text=" ".join(f"token-{index}" for index in range(221)),
            ),
        ),
    )
    embedder = _FakeEmbedder()
    writer = _FakeWriter()
    manifest_hash = hashlib.sha256(b"manifest").hexdigest()

    stage = await command.ingest_corpus(
        corpus=corpus,
        corpus_version="sample-v1",
        embedder=embedder,
        writer=writer,
        artifact_manifest_sha256=manifest_hash,
    )

    assert [chunk.position for chunk in writer.chunks] == [0, 1]
    assert [chunk.token_count for chunk in writer.chunks] == [220, 41]
    assert embedder.document_calls == [tuple(chunk.text for chunk in writer.chunks)]
    assert len(writer.embeddings) == 2
    assert stage.status == "vector_staged"
    assert stage.embedding_artifact_manifest_sha256 == manifest_hash
    assert stage.chunker_version is ChunkerVersion.TOKEN_DECODE_V1


def test_stage_manifest_write_is_complete_and_never_active(tmp_path: Path) -> None:
    target = tmp_path / "state" / "vector-stage.json"
    stage = VectorStageManifest(
        corpus_version="sample-v1",
        collection_name="sample_collection",
        chunk_count=0,
        canonical_id_checksum=hashlib.sha256(b"").hexdigest(),
        embedding_set_checksum=hashlib.sha256(b"").hexdigest(),
        embedding_artifact_manifest_sha256=hashlib.sha256(b"manifest").hexdigest(),
    )

    command.write_stage_manifest(target, stage)

    decoded = json.loads(target.read_text(encoding="utf-8"))
    assert decoded["status"] == "vector_staged"
    assert "active" not in decoded
    assert not tuple(target.parent.glob(f".{target.name}.*.tmp"))


def test_chunk_handoff_jsonl_contains_exact_vector_stage_chunks(tmp_path: Path) -> None:
    chunks = (
        Chunk(
            id="v1:chunk:fixture:0:abc",
            document_id="v1:document:fixture:one",
            corpus_version="sample-v1",
            position=0,
            text="exact graph handoff",
            token_count=3,
        ),
    )
    target = tmp_path / "state" / "chunks.jsonl"

    command.write_chunks_jsonl(target, chunks)

    assert [Chunk.model_validate_json(line) for line in target.read_text().splitlines()] == list(
        chunks
    )
    assert not tuple(target.parent.glob(f".{target.name}.*.tmp"))


def test_parser_error_is_json_and_does_not_echo_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_value = "private-corpus-value"

    exit_code = command.run(["--unexpected", raw_value])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert json.loads(captured.err)["code"] == "INVALID_REQUEST"
    assert raw_value not in captured.err
