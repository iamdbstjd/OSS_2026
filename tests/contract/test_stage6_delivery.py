"""Static delivery gates for the Stage 6 runtime surface."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ragplan.api.server import create_app

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_stage6_runtime_is_exposed_by_compose_and_documented() -> None:
    compose = yaml.safe_load((REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    environment = compose["services"]["api"]["environment"]
    required = {
        "RAGPLAN_STAGE6_MODEL_SNAPSHOT",
        "RAGPLAN_STAGE6_VECTOR_STAGE_MANIFEST",
        "RAGPLAN_STAGE6_GRAPH_STAGE_MANIFEST",
        "RAGPLAN_STAGE6_MANIFEST_ROOT",
        "RAGPLAN_STAGE6_EXTRACTOR_LOCKFILE",
        "RAGPLAN_STAGE6_QDRANT_URL",
        "RAGPLAN_STAGE6_QDRANT_COLLECTION_PREFIX",
        "RAGPLAN_STAGE6_NEO4J_URI",
        "RAGPLAN_STAGE6_NEO4J_USER",
        "RAGPLAN_STAGE6_NEO4J_DATABASE",
    }
    assert required <= set(environment)
    env_example = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")
    assert all(key in env_example for key in required)
    assert (REPOSITORY_ROOT / "scripts/search_hybrid.py").is_file()


def test_ci_runs_real_dual_store_fixed_hybrid_integration() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "RAGPLAN_TEST_QDRANT_URL" in workflow
    assert "RAGPLAN_TEST_NEO4J_URI" in workflow
    assert "tests/integration/test_fixed_hybrid.py" in workflow


def test_bilingual_readme_code_blocks_remain_identical() -> None:
    def code_blocks(path: Path) -> tuple[str, ...]:
        blocks: list[str] = []
        current: list[str] = []
        inside = False
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            if line.startswith("```"):
                current.append(line)
                if inside:
                    blocks.append("".join(current))
                    current = []
                inside = not inside
            elif inside:
                current.append(line)
        assert inside is False
        return tuple(blocks)

    assert code_blocks(REPOSITORY_ROOT / "README.md") == code_blocks(
        REPOSITORY_ROOT / "README_EN.md"
    )


def test_openapi_exposes_fixed_plan_and_fusion_provenance() -> None:
    components = create_app().openapi()["components"]["schemas"]
    assert "plan_id" in components["SearchRequest"]["properties"]
    assert "sources" in components["RetrievalHit"]["properties"]
    assert "source_contributions" in components["RetrievalHit"]["properties"]
    assert components["FusionTrace"]["properties"]["fusion_version"]["const"] == ("weighted_rrf_v1")
