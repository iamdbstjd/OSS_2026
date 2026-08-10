#!/usr/bin/env python3
"""Provision and verify the one embedding-model revision approved for Stage 3."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from huggingface_hub import snapshot_download

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.ingestion.model_manifest import (
    ModelArtifactManifest,
    load_default_model_artifact_manifest,
    load_model_artifact_manifest,
    verify_model_artifacts,
)


class _SafeArgumentParser(argparse.ArgumentParser):
    """Convert parser failures to the same redacted error envelope as runtime failures."""

    def error(self, message: str) -> None:
        del message
        raise RAGPlanError(ErrorCode.INVALID_REQUEST, "invalid command arguments")


class SnapshotDownloader(Protocol):
    """Subset of ``huggingface_hub.snapshot_download`` used by this script."""

    def __call__(
        self,
        *,
        repo_id: str,
        revision: str,
        cache_dir: str,
        local_files_only: bool,
        allow_patterns: list[str],
    ) -> str: ...


def build_parser() -> argparse.ArgumentParser:
    """Build the intentionally small, pinned-model provisioning interface."""

    parser = _SafeArgumentParser(
        description="Download and checksum-verify the pinned Stage 3 embedding model."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Dedicated Hugging Face cache directory for the approved snapshot.",
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        help="Optional artifact manifest; defaults to the packaged immutable manifest.",
    )
    return parser


def prepare_model(
    *,
    cache_dir: Path,
    manifest: ModelArtifactManifest,
    downloader: SnapshotDownloader = snapshot_download,
) -> Path:
    """Download only allowlisted files at the pinned revision and verify every checksum."""

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_root = cache_dir.resolve(strict=True)
    except OSError as exc:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "model cache directory could not be prepared",
            retryable=False,
        ) from exc

    try:
        downloaded = downloader(
            repo_id=manifest.model_id,
            revision=manifest.revision,
            cache_dir=str(cache_root),
            local_files_only=False,
            allow_patterns=sorted(manifest.artifacts),
        )
    except Exception as exc:
        raise RAGPlanError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "approved embedding-model revision could not be downloaded",
        ) from exc

    try:
        snapshot_path = Path(downloaded).resolve(strict=True)
    except OSError as exc:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "downloaded embedding-model snapshot is unavailable",
            retryable=False,
        ) from exc
    if not snapshot_path.is_relative_to(cache_root):
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "downloaded embedding-model snapshot escaped the dedicated cache",
            retryable=False,
        )

    verify_model_artifacts(snapshot_path, manifest)
    return snapshot_path


def _load_manifest(path: Path | None) -> ModelArtifactManifest:
    if path is None:
        return load_default_model_artifact_manifest()
    return load_model_artifact_manifest(path)


def _json_line(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _emit_error(error: RAGPlanError, request_id: str) -> None:
    body = error.response(request_id).model_dump(mode="json")
    print(_json_line(body), file=sys.stderr)


def run(argv: Sequence[str] | None = None) -> int:
    """Run model preparation, returning a shell-friendly status without a traceback."""

    request_id = f"prepare-model-{uuid4()}"
    try:
        args = build_parser().parse_args(argv)
        manifest = _load_manifest(args.model_manifest)
        snapshot_path = prepare_model(
            cache_dir=args.cache_dir,
            manifest=manifest,
        )
    except RAGPlanError as exc:
        _emit_error(exc, request_id)
        return 1
    except (OSError, TypeError, ValueError) as exc:
        del exc
        _emit_error(
            RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "model preparation input is invalid",
                retryable=False,
            ),
            request_id,
        )
        return 1

    print(
        _json_line(
            {
                "artifact_manifest_sha256": manifest.sha256,
                "model_id": manifest.model_id,
                "revision": manifest.revision,
                "snapshot_path": str(snapshot_path),
                "status": "ready",
            }
        )
    )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
