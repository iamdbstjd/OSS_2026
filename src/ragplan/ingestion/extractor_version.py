"""Pinned, reproducible graph-extractor provenance."""

from __future__ import annotations

import hashlib
import json
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

from ragplan.core.errors import ErrorCode, RAGPlanError

SPACY_VERSION: Final = "3.8.15"
SPACY_MODEL_PACKAGE: Final = "en-core-web-sm"
SPACY_MODEL_NAME: Final = "en_core_web_sm"
SPACY_MODEL_VERSION: Final = "3.8.0"
SPACY_MODEL_WHEEL_SHA256: Final = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
TOKENIZERS_VERSION: Final = "0.22.2"
NORMALIZATION_RULE_VERSION: Final = "entity-normalization-v1"
RELATION_RULE_VERSION: Final = "dependency-rules-v1"

_PINNED_PACKAGES: Final = {
    "spacy": SPACY_VERSION,
    SPACY_MODEL_PACKAGE: SPACY_MODEL_VERSION,
    "tokenizers": TOKENIZERS_VERSION,
}
_EXTRACTOR_SOURCE_FILES: Final = (
    "core/ids.py",
    "ingestion/entities.py",
    "ingestion/relations.py",
    "ingestion/resolver.py",
)


def verify_extractor_lockfile(path: Path) -> str:
    """Require exact extractor package versions and hashed artifacts in ``uv.lock``."""

    try:
        raw = path.read_bytes()
        payload = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "graph extractor requires a readable immutable lockfile",
            retryable=False,
        ) from exc
    packages = payload.get("package")
    if not isinstance(packages, list):
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "graph extractor lockfile has no package records",
            retryable=False,
        )
    by_name = {
        item.get("name"): item
        for item in packages
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    for package_name, expected_version in _PINNED_PACKAGES.items():
        record = by_name.get(package_name)
        if not isinstance(record, dict) or record.get("version") != expected_version:
            raise RAGPlanError(
                ErrorCode.MODEL_INCOMPATIBLE,
                "graph extractor lockfile does not match pinned package versions",
                retryable=False,
            )
        serialized = json.dumps(record, sort_keys=True, separators=(",", ":"))
        if "sha256:" not in serialized:
            raise RAGPlanError(
                ErrorCode.MODEL_INCOMPATIBLE,
                "graph extractor lockfile package is missing artifact hashes",
                retryable=False,
            )
    model_record = by_name[SPACY_MODEL_PACKAGE]
    if SPACY_MODEL_WHEEL_SHA256 not in json.dumps(model_record, sort_keys=True):
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "spaCy model wheel checksum does not match the frozen contract",
            retryable=False,
        )
    return hashlib.sha256(raw).hexdigest()


def verify_installed_extractor_packages() -> None:
    """Reject floating or mismatched runtime packages before model loading."""

    try:
        observed = {name: version(name) for name in _PINNED_PACKAGES}
    except PackageNotFoundError as exc:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "pinned graph-extraction dependencies are not installed",
            retryable=False,
        ) from exc
    if observed != _PINNED_PACKAGES:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "installed graph-extraction package versions do not match the lockfile",
            retryable=False,
        )


def extractor_source_sha256() -> str:
    """Hash normalization/rule implementation bytes used by the extractor."""

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    digest.update(NORMALIZATION_RULE_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(RELATION_RULE_VERSION.encode("utf-8"))
    for name in _EXTRACTOR_SOURCE_FILES:
        path = root / name
        try:
            source = path.read_bytes()
        except OSError as exc:
            raise RAGPlanError(
                ErrorCode.MODEL_INCOMPATIBLE,
                "graph extractor source provenance is unavailable",
                retryable=False,
            ) from exc
        digest.update(b"\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source)
    return digest.hexdigest()


def build_extractor_version(lockfile: Path | None, *, benchmark_mode: bool = True) -> str:
    """Build a version from packages, lockfile, normalization, and relation source."""

    if benchmark_mode and lockfile is None:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "benchmark graph extraction requires uv.lock provenance",
            retryable=False,
        )
    verify_installed_extractor_packages()
    lock_sha = verify_extractor_lockfile(lockfile) if lockfile is not None else "runtime-only"
    source_sha = extractor_source_sha256()
    payload = (
        f"spacy={SPACY_VERSION}|model={SPACY_MODEL_VERSION}|tokenizers={TOKENIZERS_VERSION}|"
        f"lock={lock_sha}|source={source_sha}"
    )
    return f"graph-extractor-v1-{hashlib.sha256(payload.encode()).hexdigest()}"
