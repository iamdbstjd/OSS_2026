"""Validated, deterministic plan-catalog loading and identity helpers."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ragplan.core.models import PlanDefinition

CATALOG_SCHEMA_VERSION = "v1"
DEFAULT_PLAN_CATALOG_PATH = Path(__file__).resolve().parents[3] / "configs" / "plans.yaml"
_CATALOG_FIELDS = frozenset({"schema_version", "plans"})
_PLAN_FIELDS = frozenset(
    {
        "id",
        "name",
        "vector_enabled",
        "graph_enabled",
        "vector_top_k",
        "graph_top_k",
        "graph_depth",
        "vector_weight",
        "graph_weight",
        "rerank_enabled",
        "rerank_top_k",
        "enabled_in_p0",
    }
)
_PLAN_ID_PATTERN = re.compile(r"P(?:0|[1-9][0-9]*)\Z")
_EXPECTED_PLAN_IDS = frozenset(f"P{number}" for number in range(9))


class PlanCatalogError(ValueError):
    """Raised when a plan catalog cannot meet the versioned schema contract."""


def _plan_order_key(plan_id: str) -> tuple[int, str]:
    """Return a natural, deterministic ordering key for a plan identifier."""
    return (int(plan_id[1:]), plan_id)


@dataclass(frozen=True, slots=True)
class PlanCatalog:
    """An immutable plan catalog with deterministically ordered plan definitions."""

    schema_version: str
    plans: tuple[PlanDefinition, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CATALOG_SCHEMA_VERSION:
            raise PlanCatalogError(
                f"unsupported plan catalog schema version: {self.schema_version!r}"
            )
        plan_ids = tuple(plan.id for plan in self.plans)
        if not plan_ids:
            raise PlanCatalogError("plan catalog must contain at least one plan")
        if len(plan_ids) != len(set(plan_ids)):
            raise PlanCatalogError("plan catalog contains duplicate plan IDs")
        if any(_PLAN_ID_PATTERN.fullmatch(plan_id) is None for plan_id in plan_ids):
            raise PlanCatalogError("plan catalog contains an invalid plan ID")
        if plan_ids != tuple(sorted(plan_ids, key=_plan_order_key)):
            raise PlanCatalogError("plan catalog plans must be deterministically ordered")
        if set(plan_ids) != _EXPECTED_PLAN_IDS:
            missing = sorted(_EXPECTED_PLAN_IDS - set(plan_ids), key=_plan_order_key)
            extra = sorted(set(plan_ids) - _EXPECTED_PLAN_IDS, key=_plan_order_key)
            raise PlanCatalogError(
                f"catalog v1 requires exactly P0-P8: missing={missing!r}, extra={extra!r}"
            )

    @property
    def p0_enabled_plans(self) -> tuple[PlanDefinition, ...]:
        """Return the ordered plans eligible for the P0 planning space."""
        return tuple(plan for plan in self.plans if plan.enabled_in_p0)

    def plan_for_id(self, plan_id: str) -> PlanDefinition:
        """Return the plan with ``plan_id`` or raise ``KeyError`` when absent."""
        for plan in self.plans:
            if plan.id == plan_id:
                return plan
        raise KeyError(plan_id)

    def canonical_json(self) -> str:
        """Serialize this catalog into its canonical stable JSON representation."""
        return canonical_json(
            {
                "schema_version": self.schema_version,
                "plans": [_plan_to_data(plan) for plan in self.plans],
            }
        )

    def sha256(self) -> str:
        """Return the lowercase SHA-256 identity of the canonical catalog JSON."""
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _plan_to_data(plan: PlanDefinition) -> dict[str, Any]:
    return {field: getattr(plan, field) for field in sorted(_PLAN_FIELDS)}


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        return {str(key): _canonicalize(item) for key, item in items}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return recursively key-sorted, compact, UTF-8-safe canonical JSON."""
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], context: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)!r}")
        if extra:
            details.append(f"extra={sorted(extra)!r}")
        raise PlanCatalogError(f"invalid {context} fields: {', '.join(details)}")


def load_plan_catalog(path: Path) -> PlanCatalog:
    """Load, validate, and normalize the versioned YAML plan catalog at ``path``."""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PlanCatalogError(f"unable to load plan catalog {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise PlanCatalogError("plan catalog must be a YAML mapping")
    _require_exact_keys(loaded, _CATALOG_FIELDS, "catalog")
    if loaded["schema_version"] != CATALOG_SCHEMA_VERSION:
        raise PlanCatalogError(
            f"unsupported plan catalog schema version: {loaded['schema_version']!r}"
        )
    if not isinstance(loaded["plans"], list):
        raise PlanCatalogError("plan catalog plans must be a YAML list")

    parsed_plans: list[PlanDefinition] = []
    for index, raw_plan in enumerate(loaded["plans"]):
        if not isinstance(raw_plan, dict):
            raise PlanCatalogError(f"plan at index {index} must be a YAML mapping")
        _require_exact_keys(raw_plan, _PLAN_FIELDS, f"plan at index {index}")
        plan_id = raw_plan["id"]
        if not isinstance(plan_id, str) or _PLAN_ID_PATTERN.fullmatch(plan_id) is None:
            raise PlanCatalogError(f"invalid plan ID at index {index}: {plan_id!r}")
        try:
            parsed_plans.append(PlanDefinition(**raw_plan))
        except (TypeError, ValueError) as exc:
            raise PlanCatalogError(f"invalid plan {plan_id}: {exc}") from exc

    ordered_plans = tuple(sorted(parsed_plans, key=lambda plan: _plan_order_key(plan.id)))
    return PlanCatalog(schema_version=CATALOG_SCHEMA_VERSION, plans=ordered_plans)


def load_default_plan_catalog() -> PlanCatalog:
    """Load the repository's version-controlled catalog and fail fast if invalid."""

    if DEFAULT_PLAN_CATALOG_PATH.is_file():
        return load_plan_catalog(DEFAULT_PLAN_CATALOG_PATH)
    packaged_catalog = files("ragplan").joinpath("resources", "plans.yaml")
    with as_file(packaged_catalog) as catalog_path:
        return load_plan_catalog(catalog_path)


def stable_tie_break_key(
    plan: PlanDefinition, predicted_latency_ms: float
) -> tuple[float, int, str]:
    """Order equal-quality candidates by latency, graph depth, then plan ID."""
    if not math.isfinite(predicted_latency_ms) or predicted_latency_ms < 0:
        raise ValueError("predicted latency must be finite and non-negative")
    return (predicted_latency_ms, plan.graph_depth, plan.id)
