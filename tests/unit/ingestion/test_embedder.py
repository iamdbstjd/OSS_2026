from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.ingestion.embedder import Embedder, SentenceTransformerEmbedder
from ragplan.ingestion.model_manifest import (
    EMBEDDING_DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    ModelArtifactManifest,
    load_default_model_artifact_manifest,
    load_model_artifact_manifest,
    verify_model_artifacts,
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


class _FakeTokenizer:
    def __init__(self) -> None:
        self.encode_calls: list[tuple[str, bool]] = []
        self.decode_calls: list[tuple[list[int], bool, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        return [len(part) for part in text.split()]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        self.decode_calls.append((token_ids, skip_special_tokens, clean_up_tokenization_spaces))
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeModel:
    def __init__(self, *, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.tokenizer = _FakeTokenizer()
        self.dimension = dimension
        self.encode_calls: list[tuple[list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
        self.encode_calls.append((sentences, kwargs))
        return [[3.0, 4.0, *([0.0] * (EMBEDDING_DIMENSION - 2))] for _ in sentences]


def _manifest(artifacts: dict[str, str] | None = None) -> ModelArtifactManifest:
    checksums = artifacts or {name: "0" * 64 for name in _ARTIFACT_NAMES}
    return ModelArtifactManifest(
        schema_version="v1",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        embedding_dimension=EMBEDDING_DIMENSION,
        normalize_embeddings=True,
        artifacts=checksums,
    )


def _write_snapshot(tmp_path: Path) -> tuple[Path, ModelArtifactManifest]:
    snapshot = tmp_path / "snapshot"
    artifacts: dict[str, str] = {}
    for index, name in enumerate(_ARTIFACT_NAMES):
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"artifact-{index}".encode()
        path.write_bytes(content)
        artifacts[name] = hashlib.sha256(content).hexdigest()
    return snapshot, _manifest(artifacts)


def test_manifest_loader_is_strict_and_freezes_artifacts(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "normalize_embeddings": True,
                "artifacts": {name: "a" * 64 for name in _ARTIFACT_NAMES},
            }
        ),
        encoding="utf-8",
    )

    manifest = load_model_artifact_manifest(manifest_path)

    assert manifest.revision == MODEL_REVISION
    assert isinstance(manifest.artifacts, MappingProxyType)
    assert manifest.canonical_json_bytes() == (
        json.dumps(
            {
                "artifacts": {name: "a" * 64 for name in sorted(_ARTIFACT_NAMES)},
                "embedding_dimension": EMBEDDING_DIMENSION,
                "model_id": MODEL_ID,
                "normalize_embeddings": True,
                "revision": MODEL_REVISION,
                "schema_version": "v1",
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    assert manifest.sha256 == hashlib.sha256(manifest.canonical_json_bytes()).hexdigest()
    with pytest.raises(TypeError):
        manifest.artifacts["config.json"] = "b" * 64  # type: ignore[index]


def test_checked_in_default_manifest_has_the_verified_fingerprint() -> None:
    manifest = load_default_model_artifact_manifest()

    assert manifest.sha256 == "f0635a959113bf3817d63ecd3a02b9c840795cbd17afeab9af596c9226771574"
    assert set(manifest.artifacts) == set(_ARTIFACT_NAMES)


def test_manifest_rejects_unapproved_revision_and_missing_artifacts() -> None:
    with pytest.raises(RAGPlanError) as revision_error:
        ModelArtifactManifest(
            schema_version="v1",
            model_id=MODEL_ID,
            revision="0" * 40,
            embedding_dimension=EMBEDDING_DIMENSION,
            normalize_embeddings=True,
            artifacts={name: "a" * 64 for name in _ARTIFACT_NAMES},
        )
    assert revision_error.value.code is ErrorCode.MODEL_INCOMPATIBLE

    with pytest.raises(RAGPlanError) as artifact_error:
        _manifest({"model.safetensors": "a" * 64})
    assert artifact_error.value.code is ErrorCode.MODEL_INCOMPATIBLE


def test_artifact_verification_detects_tampering_and_unverified_weights(tmp_path: Path) -> None:
    snapshot, manifest = _write_snapshot(tmp_path)
    verify_model_artifacts(snapshot, manifest)

    (snapshot / "config.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(RAGPlanError) as mismatch_error:
        verify_model_artifacts(snapshot, manifest)
    assert mismatch_error.value.code is ErrorCode.MODEL_INCOMPATIBLE

    snapshot, manifest = _write_snapshot(tmp_path / "second")
    (snapshot / "README.md").write_bytes(b"unverified")
    with pytest.raises(RAGPlanError, match="unverified artifacts"):
        verify_model_artifacts(snapshot, manifest)


@pytest.mark.asyncio
async def test_embedder_batches_documents_and_normalizes_query_once() -> None:
    model = _FakeModel()
    embedder = SentenceTransformerEmbedder(
        model=model, manifest=_manifest(), batch_size=7, executor=None
    )
    protocol_value: Embedder = embedder

    documents = await protocol_value.embed_documents(("alpha", "beta"))
    query = await protocol_value.embed_query("question")

    assert len(documents) == 2
    assert len(query) == EMBEDDING_DIMENSION
    assert math.isclose(math.sqrt(sum(value * value for value in query)), 1.0)
    assert query[:2] == pytest.approx((0.6, 0.8))
    assert [call[0] for call in model.encode_calls] == [["alpha", "beta"], ["question"]]
    assert model.encode_calls[1][1] == {
        "batch_size": 7,
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }


def test_embedder_exposes_exact_model_tokenizer_without_special_tokens() -> None:
    model = _FakeModel()
    embedder = SentenceTransformerEmbedder(model=model, manifest=_manifest(), executor=None)

    encoding = embedder.tokenizer.encode("one three")
    decoded = encoding.decode(0, 2)

    assert encoding.token_count == 2
    assert decoded == "3 5"
    assert model.tokenizer.encode_calls == [("one three", False)]
    assert model.tokenizer.decode_calls == [([3, 5], True, False)]


@pytest.mark.asyncio
async def test_embedder_rejects_model_or_output_dimension_mismatch() -> None:
    with pytest.raises(RAGPlanError) as model_error:
        SentenceTransformerEmbedder(
            model=_FakeModel(dimension=EMBEDDING_DIMENSION - 1), manifest=_manifest()
        )
    assert model_error.value.code is ErrorCode.MODEL_INCOMPATIBLE

    model = _FakeModel()

    def wrong_dimension(sentences: list[str], **kwargs: object) -> list[list[float]]:
        return [[1.0] for _ in sentences]

    model.encode = wrong_dimension  # type: ignore[method-assign]
    embedder = SentenceTransformerEmbedder(model=model, manifest=_manifest(), executor=None)
    with pytest.raises(RAGPlanError) as output_error:
        await embedder.embed_query("question")
    assert output_error.value.code is ErrorCode.MODEL_INCOMPATIBLE


def test_local_cache_resolution_is_pinned_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, manifest = _write_snapshot(tmp_path)
    model = _FakeModel()
    download_calls: list[dict[str, object]] = []
    load_calls: list[tuple[str, dict[str, object]]] = []

    def snapshot_download(**kwargs: object) -> str:
        download_calls.append(kwargs)
        return str(snapshot)

    def sentence_transformer(path: str, **kwargs: object) -> _FakeModel:
        load_calls.append((path, kwargs))
        return model

    modules = {
        "huggingface_hub": SimpleNamespace(snapshot_download=snapshot_download),
        "sentence_transformers": SimpleNamespace(SentenceTransformer=sentence_transformer),
    }
    monkeypatch.setattr(
        "ragplan.ingestion.embedder.importlib.import_module", lambda name: modules[name]
    )

    embedder = SentenceTransformerEmbedder.from_local_cache(
        manifest=manifest, cache_dir=tmp_path / "cache", device="cpu"
    )

    assert embedder.model_id == MODEL_ID
    assert embedder.manifest_sha256 == manifest.sha256
    assert download_calls == [
        {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "cache_dir": str(tmp_path / "cache"),
            "local_files_only": True,
            "allow_patterns": sorted(manifest.artifacts),
        }
    ]
    assert load_calls == [
        (
            str(snapshot),
            {"device": "cpu", "local_files_only": True, "trust_remote_code": False},
        )
    ]


@pytest.mark.asyncio
async def test_empty_document_batch_skips_model() -> None:
    model = _FakeModel()
    embedder = SentenceTransformerEmbedder(model=model, manifest=_manifest(), executor=None)

    assert await embedder.embed_documents(()) == ()
    assert model.encode_calls == []

    with pytest.raises(ValueError, match="non-empty"):
        await embedder.embed_documents(("",))
