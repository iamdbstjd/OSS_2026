from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_ci_runs_the_real_qdrant_and_pinned_model_vertical_slice() -> None:
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "scripts/prepare_model.py" in workflow
    assert "RAGPLAN_TEST_QDRANT_URL" in workflow
    assert "RAGPLAN_TEST_MODEL_SNAPSHOT" in workflow
    assert "tests/integration/test_qdrant_vector_backend.py" in workflow
    assert "tests/integration/test_vector_vertical_slice.py" in workflow


def test_stage3_runtime_mounts_are_read_only_and_configuration_is_all_or_nothing() -> None:
    compose = yaml.safe_load((REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    environment = api["environment"]

    assert {
        "RAGPLAN_STAGE3_MODEL_SNAPSHOT",
        "RAGPLAN_STAGE3_VECTOR_STAGE_MANIFEST",
        "RAGPLAN_STAGE3_QDRANT_URL",
        "RAGPLAN_STAGE3_QDRANT_COLLECTION_PREFIX",
    } <= set(environment)
    stage3_mounts = {
        volume["target"]: volume
        for volume in api["volumes"]
        if volume.get("target") in {"/opt/ragplan/models", "/opt/ragplan/artifacts"}
    }
    assert set(stage3_mounts) == {"/opt/ragplan/models", "/opt/ragplan/artifacts"}
    assert all(volume["read_only"] is True for volume in stage3_mounts.values())


def test_wheel_and_lock_configuration_include_the_pinned_cpu_embedding_runtime() -> None:
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert any(dependency.startswith("sentence-transformers") for dependency in dependencies)
    assert any(dependency.startswith("torch") for dependency in dependencies)
    assert pyproject["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cpu"}
    assert (
        force_include["configs/models/all_minilm_l6_v2.b8903db.manifest.json"]
        == "ragplan/resources/models/all_minilm_l6_v2.json"
    )
    assert (REPOSITORY_ROOT / "src/ragplan/resources/models/__init__.py").is_file()
    ignored_paths = {
        line.strip() for line in (REPOSITORY_ROOT / ".gitignore").read_text().splitlines()
    }
    assert "/models/*" in ignored_paths
    assert "!/models/.gitkeep" in ignored_paths
    assert "/artifacts/*" in ignored_paths
    assert "!/artifacts/.gitkeep" in ignored_paths
