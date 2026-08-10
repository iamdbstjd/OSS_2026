from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ragplan.planner.catalog import (
    PlanCatalogError,
    load_plan_catalog,
    stable_tie_break_key,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPOSITORY_ROOT / "configs" / "plans.yaml"
pytestmark = pytest.mark.unit


def _write_catalog(path: Path, catalog: object) -> Path:
    path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    return path


def _catalog_data() -> dict[str, object]:
    loaded = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_loads_all_plans_in_deterministic_order() -> None:
    catalog = load_plan_catalog(CATALOG_PATH)

    assert [plan.id for plan in catalog.plans] == [f"P{number}" for number in range(9)]
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
    assert catalog.plan_for_id("P5").name == "HYBRID_BALANCED"
    with pytest.raises(KeyError):
        catalog.plan_for_id("P9")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda data: data.pop("schema_version"), "missing"),
        (lambda data: data.__setitem__("schema_version", "v2"), "schema version"),
        (lambda data: data["plans"][0].pop("name"), "missing"),  # type: ignore[index]
        (lambda data: data["plans"][0].__setitem__("unexpected", True), "extra"),  # type: ignore[index]
        (lambda data: data["plans"][0].__setitem__("id", "fast"), "invalid plan ID"),  # type: ignore[index]
    ],
)
def test_rejects_schema_and_field_errors(tmp_path: Path, mutate: object, match: str) -> None:
    data = _catalog_data()
    assert callable(mutate)
    mutate(data)

    with pytest.raises(PlanCatalogError, match=match):
        load_plan_catalog(_write_catalog(tmp_path / "plans.yaml", data))


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    data = _catalog_data()
    plans = data["plans"]
    assert isinstance(plans, list)
    plans[1]["id"] = "P0"

    with pytest.raises(PlanCatalogError, match="duplicate"):
        load_plan_catalog(_write_catalog(tmp_path / "plans.yaml", data))


def test_v1_catalog_requires_every_p0_p8_definition(tmp_path: Path) -> None:
    data = _catalog_data()
    plans = data["plans"]
    assert isinstance(plans, list)
    plans.pop()

    with pytest.raises(PlanCatalogError, match="exactly P0-P8"):
        load_plan_catalog(_write_catalog(tmp_path / "plans.yaml", data))


def test_canonical_json_and_hash_ignore_yaml_plan_order(tmp_path: Path) -> None:
    data = _catalog_data()
    plans = data["plans"]
    assert isinstance(plans, list)
    reordered = {"plans": list(reversed(plans)), "schema_version": data["schema_version"]}

    original = load_plan_catalog(CATALOG_PATH)
    reordered_catalog = load_plan_catalog(_write_catalog(tmp_path / "reordered.yaml", reordered))

    assert reordered_catalog.canonical_json() == original.canonical_json()
    assert reordered_catalog.sha256() == original.sha256()
    assert reordered_catalog.sha256() == reordered_catalog.sha256().lower()


def test_stable_tie_break_prefers_latency_then_depth_then_id() -> None:
    catalog = load_plan_catalog(CATALOG_PATH)

    assert stable_tie_break_key(catalog.plan_for_id("P8"), 4.0) < stable_tie_break_key(
        catalog.plan_for_id("P5"), 5.0
    )
    assert stable_tie_break_key(catalog.plan_for_id("P5"), 5.0) < stable_tie_break_key(
        catalog.plan_for_id("P6"), 5.0
    )
    assert stable_tie_break_key(catalog.plan_for_id("P0"), 5.0) < stable_tie_break_key(
        catalog.plan_for_id("P1"), 5.0
    )
    with pytest.raises(ValueError, match="finite"):
        stable_tie_break_key(catalog.plan_for_id("P0"), float("nan"))
