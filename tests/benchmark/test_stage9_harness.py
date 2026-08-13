from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ragplan.benchmark.aggregate import (
    aggregate_run,
    percentile_type7,
    require_locked_best_fixed_for_test,
    write_aggregate_artifacts,
)
from ragplan.benchmark.config import create_run_manifest, load_benchmark_protocol
from ragplan.benchmark.contracts import SplitName
from ragplan.benchmark.records import (
    BenchmarkMethod,
    BenchmarkQueryIdentity,
    BranchTrialRecord,
    EnvironmentManifest,
    TrialObservation,
    TrialStatus,
)
from ragplan.benchmark.runner import (
    BenchmarkCase,
    BenchmarkInvocation,
    BenchmarkRunner,
    DuplicateRunError,
    RunRepository,
    load_stage9_cases,
)
from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.engine import BaselineSearchEngine, benchmark_graph_depth_2_config_version
from ragplan.core.errors import ErrorCode
from ragplan.core.models import BranchKind, BranchStatus, PlannerMode, SearchRequest
from ragplan.planner.catalog import load_default_plan_catalog

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.benchmark, pytest.mark.unit]


@pytest.fixture(scope="module")
def protocol():  # type: ignore[no-untyped-def]
    return load_benchmark_protocol()


@pytest.fixture(scope="module")
def validation_case(protocol) -> BenchmarkCase:  # type: ignore[no-untyped-def]
    cases = load_stage9_cases(
        manifest_path=REPOSITORY_ROOT / "benchmark/manifests/adaptive_rag_bench_v1.yaml",
        split_path=REPOSITORY_ROOT / "benchmark/configs/splits_v1.json",
        qrels_path=REPOSITORY_ROOT / "benchmark/qrels/qrels_v1.jsonl",
        protocol=protocol,
    )
    return next(case for case in cases if case.split.value == "validation")


def _environment(*, notes: str = "isolated fixture") -> EnvironmentManifest:
    return EnvironmentManifest(
        captured_at_utc="2026-08-13T00:00:00+00:00",
        os_name="Linux",
        os_release="fixture",
        machine="x86_64",
        cpu_model="fixture CPU",
        logical_cpu_count=1,
        cpu_governor="performance",
        python_version="3.12.13",
        qdrant_image=f"qdrant@sha256:{'a' * 64}",
        neo4j_image=f"neo4j@sha256:{'b' * 64}",
        api_image="ragplan-api:fixture",
        container_resource_limits="cpu=1,memory=1GiB",
        runtime_source_sha256="0" * 64,
        dependency_lock_sha256="1" * 64,
        docker_compose_sha256="2" * 64,
        db_tuning_sha256="3" * 64,
        notes=notes,
    )


def _repository(
    tmp_path: Path,
    *,
    protocol,  # type: ignore[no-untyped-def]
    case: BenchmarkCase,
    run_id: str = "stage9-fixture",
) -> tuple[RunRepository, object]:
    environment = _environment()
    manifest = create_run_manifest(
        run_id=run_id,
        protocol=protocol,
        environment=environment,
        query_identities=(
            BenchmarkQueryIdentity(
                query_id=case.query.query_id,
                split=case.split,
                source_dataset=case.query.source_dataset,
                query_tags=case.query.query_tags,
            ),
        ),
        created_at_utc="2026-08-13T00:00:00+00:00",
    )
    repository = RunRepository(
        tmp_path,
        protocol=protocol,
        environment=environment,
        manifest=manifest,
    )
    return repository, manifest


class _FixtureExecutor:
    def __init__(
        self,
        relevant_id: str,
        protocol,  # type: ignore[no-untyped-def]
        *,
        cancel_after: int | None = None,
    ) -> None:
        self._relevant_id = relevant_id
        self._protocol = protocol
        self._cancel_after = cancel_after
        self.calls = 0

    async def prepare_trial(self, invocation: BenchmarkInvocation) -> None:
        del invocation

    async def execute(self, invocation: BenchmarkInvocation) -> TrialObservation:
        self.calls += 1
        if self._cancel_after is not None and self.calls > self._cancel_after:
            raise asyncio.CancelledError
        method = invocation.method.method
        if method is BenchmarkMethod.GRAPH_DEPTH_2:
            return TrialObservation(
                status=TrialStatus.TIMEOUT,
                total_latency_ms=float(invocation.latency_budget_ms),
                timeout=True,
                error_code=ErrorCode.DEADLINE_EXCEEDED,
            )
        if method is BenchmarkMethod.FIXED_P6:
            return TrialObservation(
                status=TrialStatus.ERROR,
                total_latency_ms=30.0,
                error_code=ErrorCode.DEPENDENCY_UNAVAILABLE,
            )
        selected = {
            BenchmarkMethod.VECTOR: "P0",
            BenchmarkMethod.GRAPH_DEPTH_1: "P2",
            BenchmarkMethod.GRAPH_DEPTH_3: "P3",
            BenchmarkMethod.FIXED_P4: "P4",
            BenchmarkMethod.FIXED_P5: "P5",
            BenchmarkMethod.FIXED_P8: "P8",
            BenchmarkMethod.RULE: "P0",
        }[method]
        effective = (
            PlannerMode.VECTOR if method is BenchmarkMethod.RULE else invocation.method.planner
        )
        no_result = method is BenchmarkMethod.GRAPH_DEPTH_3
        partial = method is BenchmarkMethod.FIXED_P8
        latency = {
            BenchmarkMethod.FIXED_P4: 40.0,
            BenchmarkMethod.FIXED_P5: 60.0,
            BenchmarkMethod.FIXED_P8: 100.0,
        }.get(method, 25.0)
        if invocation.method.planner is PlannerMode.FIXED_HYBRID:
            branches = (
                BranchTrialRecord(
                    branch=BranchKind.VECTOR,
                    status=BranchStatus.SUCCEEDED,
                    latency_ms=latency / 2,
                ),
                BranchTrialRecord(
                    branch=BranchKind.GRAPH,
                    status=(BranchStatus.FAILED if partial else BranchStatus.SUCCEEDED),
                    latency_ms=latency / 2,
                    error_code=(ErrorCode.DEPENDENCY_UNAVAILABLE if partial else None),
                ),
            )
        else:
            branch = (
                BranchKind.GRAPH
                if invocation.method.planner is PlannerMode.GRAPH
                else BranchKind.VECTOR
            )
            branches = (
                BranchTrialRecord(
                    branch=branch,
                    status=BranchStatus.SUCCEEDED,
                    latency_ms=latency,
                ),
            )
        return TrialObservation(
            status=TrialStatus.PARTIAL if partial else TrialStatus.COMPLETE,
            ranked_chunk_ids=() if no_result else (self._relevant_id,),
            selected_plan_id=selected,
            effective_planner=effective,
            total_latency_ms=latency,
            branch_results=branches,
            fallback=partial,
            budget_violated=latency > invocation.latency_budget_ms,
            trace_config_version=(
                self._protocol.rule_runtime_config_version
                if method is BenchmarkMethod.RULE
                else (
                    benchmark_graph_depth_2_config_version(self._protocol.plan_catalog_sha256)
                    if method is BenchmarkMethod.GRAPH_DEPTH_2
                    else self._protocol.plan_catalog_sha256
                )
            ),
            trace_model_version=(
                f"{self._protocol.embedding_model_revision}:{self._protocol.extractor_version}"
            ),
            runtime_semantics_version="v1",
        )


def test_stage9_loader_uses_exact_480_train_validation_queries(protocol) -> None:  # type: ignore[no-untyped-def]
    cases = load_stage9_cases(
        manifest_path=REPOSITORY_ROOT / "benchmark/manifests/adaptive_rag_bench_v1.yaml",
        split_path=REPOSITORY_ROOT / "benchmark/configs/splits_v1.json",
        qrels_path=REPOSITORY_ROOT / "benchmark/qrels/qrels_v1.jsonl",
        protocol=protocol,
    )

    assert len(cases) == 480
    assert {case.split.value for case in cases} == {"train", "validation"}
    assert all(any(grade >= 1 for grade in case.relevance.values()) for case in cases)


@pytest.mark.asyncio
async def test_interrupted_run_resumes_without_duplicate_rows(
    tmp_path: Path,
    protocol,  # type: ignore[no-untyped-def]
    validation_case: BenchmarkCase,
) -> None:
    repository, manifest = _repository(tmp_path, protocol=protocol, case=validation_case)
    relevant_id = next(iter(validation_case.relevance))
    interrupted = _FixtureExecutor(relevant_id, protocol, cancel_after=17)
    runner = BenchmarkRunner(
        protocol=protocol,
        run_manifest=manifest,  # type: ignore[arg-type]
        cases=(validation_case,),
        executor=interrupted,
        repository=repository,
    )

    with pytest.raises(asyncio.CancelledError):
        await runner.run()
    assert len(repository.load_records()) == 17

    resumed = BenchmarkRunner(
        protocol=protocol,
        run_manifest=manifest,  # type: ignore[arg-type]
        cases=(validation_case,),
        executor=_FixtureExecutor(relevant_id, protocol),
        repository=repository,
    )
    summary = await resumed.run()

    assert summary.complete
    assert summary.preexisting_rows == 17
    assert summary.total_rows == 1 * 9 * 4 * 13
    records = repository.load_records()
    assert len({record.trial_id for record in records}) == len(records)


@pytest.mark.asyncio
async def test_aggregation_keeps_failures_in_denominators_and_is_byte_stable(
    tmp_path: Path,
    protocol,  # type: ignore[no-untyped-def]
    validation_case: BenchmarkCase,
) -> None:
    repository, manifest = _repository(tmp_path, protocol=protocol, case=validation_case)
    executor = _FixtureExecutor(next(iter(validation_case.relevance)), protocol)
    await BenchmarkRunner(
        protocol=protocol,
        run_manifest=manifest,  # type: ignore[arg-type]
        cases=(validation_case,),
        executor=executor,
        repository=repository,
    ).run()
    records = repository.load_records()
    report = aggregate_run(records, manifest=manifest)  # type: ignore[arg-type]

    assert report.raw_record_count == 468
    assert {item.plan_id for item in report.best_fixed.selections} == {"P4"}
    timeout = next(
        item
        for item in report.summaries
        if item.phase == "measured"
        and item.dimension.value == "overall"
        and item.method is BenchmarkMethod.GRAPH_DEPTH_2
        and item.latency_budget_ms == 50
    )
    error = next(
        item
        for item in report.summaries
        if item.phase == "measured"
        and item.dimension.value == "overall"
        and item.method is BenchmarkMethod.FIXED_P6
        and item.latency_budget_ms == 50
    )
    assert timeout.trial_count == 10
    assert timeout.timeout_rate == 1.0
    assert timeout.recall_at_10 == 0.0
    assert error.error_rate == 1.0

    write_aggregate_artifacts(repository.run_dir, records=records, report=report)
    first = {
        name: (repository.run_dir / name).read_bytes()
        for name in ("aggregate.json", "aggregate.csv", "raw_trials.csv", "checksums.json")
    }
    write_aggregate_artifacts(repository.run_dir, records=records, report=report)
    assert first == {name: (repository.run_dir / name).read_bytes() for name in first}

    with pytest.raises(ValueError, match="validation lock"):
        require_locked_best_fixed_for_test(
            report.best_fixed,
            {50: "P5", 100: "P4", 200: "P4", 500: "P4"},
        )


@pytest.mark.asyncio
async def test_incomplete_or_version_mismatched_raw_matrix_is_rejected(
    tmp_path: Path,
    protocol,  # type: ignore[no-untyped-def]
    validation_case: BenchmarkCase,
) -> None:
    repository, manifest = _repository(tmp_path, protocol=protocol, case=validation_case)
    executor = _FixtureExecutor(next(iter(validation_case.relevance)), protocol)
    await BenchmarkRunner(
        protocol=protocol,
        run_manifest=manifest,  # type: ignore[arg-type]
        cases=(validation_case,),
        executor=executor,
        repository=repository,
    ).run()
    records = repository.load_records()

    with pytest.raises(ValueError, match="incomplete"):
        aggregate_run(records[:-1], manifest=manifest)  # type: ignore[arg-type]
    changed = records[0].model_copy(update={"environment_manifest_sha256": "f" * 64})
    with pytest.raises(ValueError, match="versions"):
        aggregate_run((changed, *records[1:]), manifest=manifest)  # type: ignore[arg-type]
    changed_split = records[0].model_copy(update={"split": SplitName.TRAIN})
    with pytest.raises(ValueError, match="identity"):
        aggregate_run((changed_split, *records[1:]), manifest=manifest)  # type: ignore[arg-type]


def test_duplicate_run_identity_rejects_changed_environment(
    tmp_path: Path,
    protocol,  # type: ignore[no-untyped-def]
    validation_case: BenchmarkCase,
) -> None:
    repository, manifest = _repository(tmp_path, protocol=protocol, case=validation_case)
    assert repository.run_dir.is_dir()

    with pytest.raises(DuplicateRunError, match="environment"):
        RunRepository(
            tmp_path,
            protocol=protocol,
            environment=_environment(notes="different environment"),
            manifest=manifest,  # type: ignore[arg-type]
        )

    with repository.exclusive_lock(), pytest.raises(DuplicateRunError, match="already"):
        RunRepository(
            tmp_path,
            protocol=protocol,
            environment=_environment(),
            manifest=manifest,  # type: ignore[arg-type]
        )


def test_duplicate_query_cases_are_rejected(
    tmp_path: Path,
    protocol,  # type: ignore[no-untyped-def]
    validation_case: BenchmarkCase,
) -> None:
    repository, manifest = _repository(tmp_path, protocol=protocol, case=validation_case)

    with pytest.raises(ValueError, match="duplicate query"):
        BenchmarkRunner(
            protocol=protocol,
            run_manifest=manifest,  # type: ignore[arg-type]
            cases=(validation_case, validation_case),
            executor=_FixtureExecutor(next(iter(validation_case.relevance)), protocol),
            repository=repository,
        )


def test_type7_percentile_fixture() -> None:
    assert percentile_type7([1.0, 2.0, 3.0, 4.0], 0.50) == 2.5
    assert percentile_type7([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_graph_depth_two_is_an_internal_traceable_benchmark_plan() -> None:
    engine = object.__new__(BaselineSearchEngine)
    catalog = load_default_plan_catalog()
    engine._plan_catalog = catalog  # noqa: SLF001
    deadline = Deadline.start(500, clock=ManualClock())
    request = SearchRequest(
        query="Which relation connects these entities?",
        top_k=10,
        latency_budget_ms=500,
        planner=PlannerMode.GRAPH,
    )

    decision = engine._select_graph_plan(  # noqa: SLF001
        request,
        deadline,
        graph_depth_override=2,
    )

    assert decision.selected_plan_id == "P3"
    assert decision.selected_plan is not None
    assert decision.selected_plan.graph_depth == 2
    assert decision.config_version != catalog.sha256()
