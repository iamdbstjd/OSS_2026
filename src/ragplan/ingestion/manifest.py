"""Durable ingestion-run manifests and the single served-corpus pointer."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import FrozenModel, IngestionManifest, NonEmptyString, Sha256Hex


class ActiveCorpusPointer(FrozenModel):
    schema_version: Literal["v1"] = "v1"
    corpus_version: NonEmptyString
    ingestion_run_id: NonEmptyString
    ingestion_manifest_sha256: Sha256Hex


def ingestion_manifest_sha256(manifest: IngestionManifest) -> str:
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ManifestRepository:
    """Immutable run records plus an atomic active pointer on local storage."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._runs = root / "runs"
        self._active = root / "active_corpus.json"

    @property
    def active_pointer_path(self) -> Path:
        return self._active

    def record(self, manifest: IngestionManifest) -> Path:
        path = self._run_path(manifest.ingestion_run_id)
        serialized = _canonical_json_bytes(manifest.model_dump(mode="json"))
        if path.exists():
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise _manifest_io_error() from exc
            if existing != serialized:
                raise RAGPlanError(
                    ErrorCode.CORPUS_INCONSISTENT,
                    "an ingestion run ID cannot be reused with different evidence",
                    retryable=False,
                )
            return path
        _atomic_write(path, serialized)
        return path

    def activate(self, manifest: IngestionManifest) -> ActiveCorpusPointer:
        """Record an already-reconciled manifest, then atomically switch serving."""

        if manifest.activation_status.value != "active":
            raise ValueError("only an active reconciled ingestion manifest can be served")
        self.record(manifest)
        pointer = ActiveCorpusPointer(
            corpus_version=manifest.corpus_version,
            ingestion_run_id=manifest.ingestion_run_id,
            ingestion_manifest_sha256=ingestion_manifest_sha256(manifest),
        )
        _atomic_write(
            self._active,
            _canonical_json_bytes(pointer.model_dump(mode="json")),
        )
        return pointer

    def load_run(self, ingestion_run_id: str) -> IngestionManifest:
        return _load_model(self._run_path(ingestion_run_id), IngestionManifest)

    def load_active(self) -> tuple[ActiveCorpusPointer, IngestionManifest]:
        pointer = _load_model(self._active, ActiveCorpusPointer)
        manifest = self.load_run(pointer.ingestion_run_id)
        if (
            manifest.corpus_version != pointer.corpus_version
            or ingestion_manifest_sha256(manifest) != pointer.ingestion_manifest_sha256
            or manifest.activation_status.value != "active"
        ):
            raise RAGPlanError(
                ErrorCode.CORPUS_INCONSISTENT,
                "active corpus pointer does not match its immutable ingestion manifest",
            )
        return pointer, manifest

    def rollback(self, ingestion_run_id: str) -> ActiveCorpusPointer:
        """Atomically point serving at a previously reconciled active run."""

        return self.activate(self.load_run(ingestion_run_id))

    def discard(self, ingestion_run_id: str) -> None:
        """Discard one explicit non-served run record; storage cleanup is separate."""

        try:
            pointer, _ = self.load_active()
        except RAGPlanError as exc:
            if exc.code is not ErrorCode.NOT_READY:
                raise
        else:
            if pointer.ingestion_run_id == ingestion_run_id:
                raise RAGPlanError(
                    ErrorCode.CORPUS_INCONSISTENT,
                    "the currently served ingestion run cannot be discarded",
                    retryable=False,
                )
        path = self._run_path(ingestion_run_id)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise _manifest_io_error() from exc

    def _run_path(self, ingestion_run_id: str) -> Path:
        if not isinstance(ingestion_run_id, str) or not ingestion_run_id.strip():
            raise ValueError("ingestion_run_id must not be blank")
        filename = hashlib.sha256(ingestion_run_id.encode("utf-8")).hexdigest()
        return self._runs / f"{filename}.json"


def write_contract_json(path: Path, model: FrozenModel) -> None:
    """Atomically serialize one frozen contract as canonical UTF-8 JSON."""

    _atomic_write(path, _canonical_json_bytes(model.model_dump(mode="json")))


def load_contract_json[ModelT: FrozenModel](path: Path, model: type[ModelT]) -> ModelT:
    """Load strict JSON with duplicate-key detection and JSON-aware enum parsing."""

    return _load_model(path, model)


class ActiveCorpusResolver:
    """Resolve only the atomically activated dual-store corpus."""

    def __init__(self, repository: ManifestRepository) -> None:
        self._repository = repository

    def resolve(self) -> str:
        try:
            _, manifest = self._repository.load_active()
        except RAGPlanError as exc:
            if exc.code is ErrorCode.NOT_READY:
                raise
            raise
        return manifest.corpus_version


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            os.fchmod(output.fileno(), 0o644)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise _manifest_io_error() from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate JSON key")
        payload[key] = value
    return payload


def _load_model[ModelT: FrozenModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
        canonical_payload = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return model.model_validate_json(canonical_payload)
    except FileNotFoundError as exc:
        raise RAGPlanError(
            ErrorCode.NOT_READY,
            "no active corpus has been reconciled",
        ) from exc
    except RAGPlanError:
        raise
    except Exception as exc:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "ingestion manifest is invalid",
            retryable=False,
        ) from exc


def _manifest_io_error() -> RAGPlanError:
    return RAGPlanError(
        ErrorCode.DEPENDENCY_UNAVAILABLE,
        "ingestion manifest storage is unavailable",
    )
