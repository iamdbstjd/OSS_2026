from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.ingestion.model_manifest import (
    EMBEDDING_DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    ModelArtifactManifest,
)

pytestmark = pytest.mark.unit

_ARTIFACT_NAMES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def _load_command() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "prepare_model.py"
    spec = importlib.util.spec_from_file_location("ragplan_script_prepare_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


command = _load_command()


def _snapshot(cache_dir: Path) -> tuple[Path, ModelArtifactManifest]:
    path = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "rev"
    artifacts: dict[str, str] = {}
    for index, name in enumerate(_ARTIFACT_NAMES):
        artifact = path / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        content = f"artifact-{index}".encode()
        artifact.write_bytes(content)
        artifacts[name] = hashlib.sha256(content).hexdigest()
    return path, ModelArtifactManifest(
        schema_version="v1",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        embedding_dimension=EMBEDDING_DIMENSION,
        normalize_embeddings=True,
        artifacts=artifacts,
    )


def test_prepare_model_pins_revision_allowlists_and_verifies(tmp_path: Path) -> None:
    cache_dir = tmp_path / "dedicated-cache"
    snapshot, manifest = _snapshot(cache_dir)
    calls: list[dict[str, object]] = []

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    prepared = command.prepare_model(
        cache_dir=cache_dir,
        manifest=manifest,
        downloader=downloader,
    )

    assert prepared == snapshot.resolve()
    assert calls == [
        {
            "allow_patterns": sorted(_ARTIFACT_NAMES),
            "cache_dir": str(cache_dir.resolve()),
            "local_files_only": False,
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
        }
    ]


def test_prepare_model_rejects_a_snapshot_outside_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    snapshot, manifest = _snapshot(tmp_path / "other")

    with pytest.raises(RAGPlanError) as captured:
        command.prepare_model(
            cache_dir=cache_dir,
            manifest=manifest,
            downloader=lambda **kwargs: str(snapshot),
        )

    assert captured.value.code is ErrorCode.MODEL_INCOMPATIBLE


def test_parser_failure_uses_stable_redacted_error_body(capsys: pytest.CaptureFixture[str]) -> None:
    secret_path = "/tmp/private-model-location"

    exit_code = command.run(["--unknown", secret_path])

    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert error["code"] == "INVALID_REQUEST"
    assert error["retryable"] is False
    assert secret_path not in captured.err
