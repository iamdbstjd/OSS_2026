"""Strict Stage 2 benchmark artifact contracts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self
from urllib.parse import quote

from pydantic import Field, StringConstraints, model_validator

from ragplan.core.models import FrozenModel, NonEmptyString, Sha256Hex

BENCHMARK_ID = "adaptive_rag_bench_v1"
BENCHMARK_SCHEMA_VERSION = "v1"
CORPUS_VERSION: Final[Literal["adaptive_rag_bench_v1-corpus-v1"]] = (
    "adaptive_rag_bench_v1-corpus-v1"
)
QRELS_VERSION: Final[Literal["qrels_v1"]] = "qrels_v1"
SPLIT_SEED: Final[Literal[20260809]] = 20260809
SELECTION_RULE: Final[
    Literal["SHA256(source_dataset + ':' + source_query_id + ':20260809') ascending"]
] = "SHA256(source_dataset + ':' + source_query_id + ':20260809') ascending"

QueryId = Annotated[
    str,
    StringConstraints(
        pattern=r"^adaptive_rag_bench_v1:(?:nq|hotpot_bridge|hotpot_comparison|musique):.+$"
    ),
]
CanonicalChunkId = Annotated[str, StringConstraints(pattern=r"^v1:chunk:.+$")]
CanonicalDocumentId = Annotated[str, StringConstraints(pattern=r"^v1:document:.+$")]


class SourceDataset(StrEnum):
    NQ = "nq"
    HOTPOT_BRIDGE = "hotpot_bridge"
    HOTPOT_COMPARISON = "hotpot_comparison"
    MUSIQUE = "musique"


class QueryTag(StrEnum):
    SEMANTIC = "semantic"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    COMPARISON = "comparison"
    TWO_HOP = "2hop"
    THREE_HOP = "3hop"


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


SOURCE_TOTALS: Mapping[SourceDataset, int] = {
    SourceDataset.NQ: 200,
    SourceDataset.HOTPOT_BRIDGE: 200,
    SourceDataset.HOTPOT_COMPARISON: 100,
    SourceDataset.MUSIQUE: 100,
}
SOURCE_SPLIT_QUOTAS: Mapping[SourceDataset, Mapping[SplitName, int]] = {
    SourceDataset.NQ: {
        SplitName.TRAIN: 120,
        SplitName.VALIDATION: 40,
        SplitName.TEST: 40,
    },
    SourceDataset.HOTPOT_BRIDGE: {
        SplitName.TRAIN: 120,
        SplitName.VALIDATION: 40,
        SplitName.TEST: 40,
    },
    SourceDataset.HOTPOT_COMPARISON: {
        SplitName.TRAIN: 60,
        SplitName.VALIDATION: 20,
        SplitName.TEST: 20,
    },
    SourceDataset.MUSIQUE: {
        SplitName.TRAIN: 60,
        SplitName.VALIDATION: 20,
        SplitName.TEST: 20,
    },
}
SPLIT_TOTALS: Mapping[SplitName, int] = {
    SplitName.TRAIN: 360,
    SplitName.VALIDATION: 120,
    SplitName.TEST: 120,
}
REQUIRED_TEST_TAG_MINIMUM = 15


class TagEvidence(FrozenModel):
    tag: QueryTag
    rule: NonEmptyString
    evidence: NonEmptyString


class BenchmarkQuery(FrozenModel):
    query_id: QueryId
    source_dataset: SourceDataset
    source_query_id: NonEmptyString
    question: NonEmptyString
    answers: tuple[NonEmptyString, ...] = Field(min_length=1)
    query_tags: tuple[QueryTag, ...] = Field(min_length=1)
    tag_evidence: tuple[TagEvidence, ...] = Field(min_length=1)
    selection_hash: Sha256Hex
    supporting_document_ids: tuple[CanonicalDocumentId, ...] = Field(min_length=1)
    group_keys: tuple[NonEmptyString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent_tags(self) -> Self:
        if len(set(self.query_tags)) != len(self.query_tags):
            raise ValueError("query_tags must be unique")
        evidence_tags = tuple(item.tag for item in self.tag_evidence)
        if set(evidence_tags) != set(self.query_tags) or len(set(evidence_tags)) != len(
            evidence_tags
        ):
            raise ValueError("tag_evidence must cover every query tag exactly once")
        if len(set(self.supporting_document_ids)) != len(self.supporting_document_ids):
            raise ValueError("supporting_document_ids must be unique")
        if len(set(self.group_keys)) != len(self.group_keys):
            raise ValueError("group_keys must be unique")
        expected_query_id = (
            f"{BENCHMARK_ID}:{self.source_dataset.value}:{quote(self.source_query_id, safe='-._~')}"
        )
        if self.query_id != expected_query_id:
            raise ValueError("query_id does not match source_dataset/source_query_id")
        expected_selection_hash = hashlib.sha256(
            f"{self.source_dataset.value}:{self.source_query_id}:{SPLIT_SEED}".encode()
        ).hexdigest()
        if self.selection_hash != expected_selection_hash:
            raise ValueError("selection_hash does not match the frozen selection rule")
        required_document_groups = {
            f"document:{document_id}" for document_id in self.supporting_document_ids
        }
        if not required_document_groups <= set(self.group_keys):
            raise ValueError("every supporting document must be a split group key")
        if sum(key.startswith("template:") for key in self.group_keys) != 1:
            raise ValueError("each query must have exactly one template group key")
        return self


class BenchmarkManifest(FrozenModel):
    schema_version: Annotated[str, StringConstraints(pattern=r"^v1$")] = BENCHMARK_SCHEMA_VERSION
    benchmark_id: Annotated[str, StringConstraints(pattern=r"^adaptive_rag_bench_v1$")] = (
        BENCHMARK_ID
    )
    selection_seed: Literal[20260809] = SPLIT_SEED
    selection_rule: Literal[
        "SHA256(source_dataset + ':' + source_query_id + ':20260809') ascending"
    ] = SELECTION_RULE
    corpus_version: Literal["adaptive_rag_bench_v1-corpus-v1"] = CORPUS_VERSION
    qrels_version: Literal["qrels_v1"] = QRELS_VERSION
    queries: tuple[BenchmarkQuery, ...] = Field(min_length=600, max_length=600)
    manifest_sha256: Sha256Hex

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        ids = tuple(item.query_id for item in self.queries)
        if len(set(ids)) != 600:
            raise ValueError("benchmark query IDs must be unique")
        if ids != tuple(sorted(ids)):
            raise ValueError("benchmark queries must be sorted by query_id")
        counts = Counter(item.source_dataset for item in self.queries)
        if counts != Counter(SOURCE_TOTALS):
            raise ValueError("benchmark source counts do not match the frozen contract")
        expected = benchmark_manifest_sha256(self.queries)
        if self.manifest_sha256 != expected:
            raise ValueError("manifest_sha256 does not match the canonical query payload")
        return self


class SplitAssignment(FrozenModel):
    query_id: QueryId
    split: SplitName


class SplitManifest(FrozenModel):
    schema_version: Annotated[str, StringConstraints(pattern=r"^v1$")] = BENCHMARK_SCHEMA_VERSION
    benchmark_id: Annotated[str, StringConstraints(pattern=r"^adaptive_rag_bench_v1$")] = (
        BENCHMARK_ID
    )
    split_seed: Literal[20260809] = SPLIT_SEED
    benchmark_manifest_sha256: Sha256Hex
    assignments: tuple[SplitAssignment, ...] = Field(min_length=600, max_length=600)
    split_hash: Sha256Hex

    @model_validator(mode="after")
    def _validate_assignments(self) -> Self:
        ids = tuple(item.query_id for item in self.assignments)
        if len(set(ids)) != 600:
            raise ValueError("each query must have exactly one split assignment")
        if ids != tuple(sorted(ids)):
            raise ValueError("split assignments must be sorted by query_id")
        counts = Counter(item.split for item in self.assignments)
        if counts != Counter(SPLIT_TOTALS):
            raise ValueError("split totals do not match 360/120/120")
        if self.split_hash != split_manifest_sha256(self.assignments):
            raise ValueError("split_hash does not match the canonical assignments")
        return self


class Qrel(FrozenModel):
    query_id: QueryId
    canonical_chunk_id: CanonicalChunkId
    relevance_grade: int = Field(ge=0, le=2)
    query_tags: tuple[QueryTag, ...] = Field(min_length=1)
    source_dataset: SourceDataset
    supporting_fact_id: NonEmptyString


class CorpusDocumentIndex(FrozenModel):
    document_id: CanonicalDocumentId
    source_dataset: SourceDataset
    source_document_id: NonEmptyString
    title: NonEmptyString
    text_sha256: Sha256Hex
    query_ids: tuple[QueryId, ...] = Field(min_length=1)
    supporting_query_ids: tuple[QueryId, ...] = ()


class CorpusChunkIndex(FrozenModel):
    canonical_chunk_id: CanonicalChunkId
    document_id: CanonicalDocumentId
    position: int = Field(ge=0)
    token_count: int = Field(ge=1, le=220)
    text_sha256: Sha256Hex


class ImmutableTestManifest(FrozenModel):
    schema_version: Annotated[str, StringConstraints(pattern=r"^v1$")] = BENCHMARK_SCHEMA_VERSION
    benchmark_id: Annotated[str, StringConstraints(pattern=r"^adaptive_rag_bench_v1$")] = (
        BENCHMARK_ID
    )
    split_hash: Sha256Hex
    query_ids: tuple[QueryId, ...] = Field(min_length=120, max_length=120)
    query_ids_sha256: Sha256Hex

    @model_validator(mode="after")
    def _validate_test_ids(self) -> Self:
        if len(set(self.query_ids)) != 120:
            raise ValueError("test query IDs must be unique")
        if self.query_ids_sha256 != canonical_sha256(list(self.query_ids)):
            raise ValueError("query_ids_sha256 does not match the frozen test IDs")
        if self.query_ids != tuple(sorted(self.query_ids)):
            raise ValueError("immutable test query IDs must be sorted")
        return self


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible benchmark state with one canonical encoding."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def benchmark_manifest_sha256(queries: Sequence[BenchmarkQuery]) -> str:
    payload = [item.model_dump(mode="json") for item in queries]
    return canonical_sha256(payload)


def split_manifest_sha256(assignments: Iterable[SplitAssignment]) -> str:
    payload = [item.model_dump(mode="json") for item in assignments]
    return canonical_sha256(payload)
