from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ragplan.api.server import create_app
from ragplan.planner.catalog import PlanCatalogError, load_plan_catalog

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_plan_catalog_matches_the_p0_p8_addendum_contract() -> None:
    catalog = load_plan_catalog(REPOSITORY_ROOT / "configs" / "plans.yaml")

    actual = {
        plan.id: (
            plan.name,
            plan.vector_enabled,
            plan.graph_enabled,
            plan.vector_top_k,
            plan.graph_top_k,
            plan.graph_depth,
            plan.vector_weight,
            plan.graph_weight,
            plan.rerank_enabled,
            plan.rerank_top_k,
            plan.enabled_in_p0,
        )
        for plan in catalog.plans
    }
    assert actual == {
        "P0": ("VECTOR_FAST", True, False, 10, 0, 0, 1.0, 0.0, False, 0, True),
        "P1": ("VECTOR_WIDE", True, False, 30, 0, 0, 1.0, 0.0, False, 0, True),
        "P2": ("GRAPH_SHALLOW", False, True, 0, 20, 1, 0.0, 1.0, False, 0, True),
        "P3": ("GRAPH_DEEP", False, True, 0, 30, 3, 0.0, 1.0, False, 0, True),
        "P4": ("HYBRID_VECTOR_HEAVY", True, True, 20, 15, 1, 0.7, 0.3, False, 0, True),
        "P5": ("HYBRID_BALANCED", True, True, 20, 20, 1, 0.5, 0.5, False, 0, True),
        "P6": ("HYBRID_GRAPH_HEAVY", True, True, 15, 30, 2, 0.3, 0.7, False, 0, True),
        "P7": ("HYBRID_RERANK", True, True, 30, 30, 2, 0.5, 0.5, True, 10, False),
        "P8": ("HYBRID_GRAPH_DEEP", True, True, 15, 40, 3, 0.25, 0.75, False, 0, True),
    }
    assert all(plan.vector_enabled or plan.graph_enabled for plan in catalog.plans)
    assert [plan.id for plan in catalog.p0_enabled_plans] == [
        "P0",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P8",
    ]
    assert catalog.sha256() == "8abde227cfe940a3c1467e22e5421355597e53116782f23f6f9e07d464e155d2"


def test_invalid_catalog_fails_application_creation(tmp_path: Path) -> None:
    catalog_path = tmp_path / "plans.yaml"
    catalog_path.write_text(
        yaml.safe_dump({"schema_version": "v1", "plans": []}),
        encoding="utf-8",
    )

    with pytest.raises(PlanCatalogError, match="at least one"):
        create_app(plan_catalog_path=catalog_path)
