"""Deterministic, atomic Stage 2 artifact persistence and loading."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ragplan.benchmark.contracts import (
    BenchmarkManifest,
    CorpusChunkIndex,
    CorpusDocumentIndex,
    ImmutableTestManifest,
    Qrel,
    SplitManifest,
    canonical_json_bytes,
)
from ragplan.core.models import Chunk, FrozenModel


def write_yaml_model(path: Path, model: FrozenModel) -> None:
    payload = yaml.safe_dump(
        model.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")
    _atomic_write(path, payload)


def write_json_model(path: Path, model: FrozenModel) -> None:
    payload = (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write(path, payload)


def write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    _atomic_write(path, payload)


def write_bytes(path: Path, payload: bytes) -> None:
    """Atomically persist an already-canonicalized benchmark artifact."""

    _atomic_write(path, payload)


def write_jsonl_models(path: Path, models: Iterable[FrozenModel]) -> None:
    rows = (canonical_json_bytes(model.model_dump(mode="json")) for model in models)
    _atomic_write(path, b"\n".join(rows) + b"\n")


def write_jsonl_values(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    rows = (canonical_json_bytes(dict(value)) for value in values)
    _atomic_write(path, b"\n".join(rows) + b"\n")


def load_benchmark_manifest(path: Path) -> BenchmarkManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return BenchmarkManifest.model_validate(payload, strict=False)


def load_split_manifest(path: Path) -> SplitManifest:
    return SplitManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_immutable_test_manifest(path: Path) -> ImmutableTestManifest:
    return ImmutableTestManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_qrels(path: Path) -> tuple[Qrel, ...]:
    return tuple(Qrel.model_validate_json(line) for line in _nonempty_lines(path))


def load_corpus_index(path: Path) -> tuple[CorpusDocumentIndex, ...]:
    return tuple(CorpusDocumentIndex.model_validate_json(line) for line in _nonempty_lines(path))


def load_chunk_index(path: Path) -> tuple[CorpusChunkIndex, ...]:
    return tuple(CorpusChunkIndex.model_validate_json(line) for line in _nonempty_lines(path))


def load_chunks(path: Path) -> tuple[Chunk, ...]:
    return tuple(Chunk.model_validate_json(line) for line in _nonempty_lines(path))


def _nonempty_lines(path: Path) -> Iterable[str]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                yield line
            elif line_number == 1:
                raise ValueError(f"artifact starts with an empty row: {path}")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
