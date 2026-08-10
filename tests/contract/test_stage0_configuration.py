from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_default_configuration_is_local_safe_and_rule_first() -> None:
    config_path = REPOSITORY_ROOT / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["schema_version"] == "v1"
    assert config["server"] == {"host": "127.0.0.1", "port": 8000}
    assert config["planner"]["mode"] == "rule"
    assert config["logging"]["mode"] == "redacted"
    assert "password" not in config["graph"]


def test_environment_example_marks_demo_secret_and_is_ignored() -> None:
    example = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")
    gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "local/demo" in example
    assert "RAGPLAN_GRAPH__PASSWORD=ragplan-demo-change-me" in example
    assert ".env" in {line.strip() for line in gitignore.splitlines()}


def test_compose_and_dockerfile_do_not_use_latest_tags() -> None:
    compose_path = REPOSITORY_ROOT / "docker-compose.yml"
    compose = compose_path.read_text(encoding="utf-8")
    dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert ":latest" not in compose
    assert ":latest" not in dockerfile
    assert "NEO4J_PASSWORD:?" in compose
    assert "NEO4J_PASSWORD:-" not in compose

    compose_config = yaml.safe_load(compose)
    for service_name in ("qdrant", "neo4j"):
        image = compose_config["services"][service_name]["image"]
        assert re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", image)

    assert re.search(r"^FROM python:3\.12\.13-slim-bookworm@sha256:[0-9a-f]{64}$", dockerfile, re.M)
    assert "USER ragplan" in dockerfile


def test_package_metadata_and_lock_cover_stage_zero_groups() -> None:
    pyproject_path = REPOSITORY_ROOT / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "ragplan"
    assert pyproject["project"]["requires-python"] == ">=3.12,<3.13"
    assert pyproject["project"]["scripts"]["ragplan"] == "ragplan.cli.app:main"
    assert set(pyproject["dependency-groups"]) == {
        "dev",
        "benchmark",
        "graph-extraction",
    }
    assert (REPOSITORY_ROOT / "uv.lock").is_file()
