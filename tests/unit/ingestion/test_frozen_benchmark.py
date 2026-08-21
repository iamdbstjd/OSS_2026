from __future__ import annotations

from pathlib import Path

import pytest

from ragplan.benchmark.config import load_benchmark_protocol
from ragplan.core.errors import RAGPlanError
from ragplan.ingestion.frozen_benchmark import load_verified_frozen_chunks

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


def test_frozen_chunks_match_the_stage2_index_and_protocol() -> None:
    chunks = load_verified_frozen_chunks(
        ROOT / "benchmark/datasets/normalized/chunks_v1.jsonl",
        ROOT / "benchmark/manifests/chunk_index_v1.jsonl",
        protocol=load_benchmark_protocol(),
    )

    assert len(chunks) == 8604
    assert chunks == tuple(sorted(chunks, key=lambda item: item.canonical_chunk_id))


def test_frozen_chunk_loader_rejects_a_missing_index(tmp_path: Path) -> None:
    with pytest.raises(RAGPlanError, match="chunks or index"):
        load_verified_frozen_chunks(
            ROOT / "benchmark/datasets/normalized/chunks_v1.jsonl",
            tmp_path / "missing.jsonl",
            protocol=load_benchmark_protocol(),
        )
