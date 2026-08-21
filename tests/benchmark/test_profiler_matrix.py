from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ragplan.benchmark.config import load_benchmark_protocol
from ragplan.benchmark.contracts import SplitName
from ragplan.benchmark.profile_records import (
    ProfileRunManifest,
    create_profile_protocol,
    create_profile_run_manifest,
)
from ragplan.benchmark.profiler import (
    PlanProfiler,
    ProfileInvocation,
    ProfileRepository,
    ProfileTrialObservation,
    build_training_matrix,
    write_training_matrix_artifacts,
)
from ragplan.benchmark.records import (
    BenchmarkQueryIdentity,
    BranchTrialRecord,
    EnvironmentManifest,
    TrialStatus,
)
from ragplan.benchmark.runner import BenchmarkCase, RawEvidenceError, load_stage9_cases
from ragplan.core.errors import ErrorCode
from ragplan.core.models import BranchKind, BranchStatus, QueryFeatures
from ragplan.planner.catalog import load_default_plan_catalog

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.benchmark, pytest.mark.unit]


def _environment() -> EnvironmentManifest:
    return EnvironmentManifest(
        captured_at_utc="2026-08-19T00:00:00+00:00",
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
        notes="isolated Stage 10 fixture",
    )


@pytest.fixture(scope="module")
def profile_fixture() -> tuple[object, object, tuple[BenchmarkCase, ...]]:
    baseline = load_benchmark_protocol()
    cases = load_stage9_cases(
        manifest_path=REPOSITORY_ROOT / "benchmark/manifests/adaptive_rag_bench_v1.yaml",
        split_path=REPOSITORY_ROOT / "benchmark/configs/splits_v1.json",
        qrels_path=REPOSITORY_ROOT / "benchmark/qrels/qrels_v1.jsonl",
        protocol=baseline,
    )[:3]
    protocol = create_profile_protocol(
        baseline,
        load_default_plan_catalog(),
        plan_ids=("P0", "P2", "P4"),
        latency_budgets_ms=(100,),
    )
    manifest = create_profile_run_manifest(
        run_id="stage10-fixture",
        protocol=protocol,
        environment=_environment(),
        query_identities=tuple(
            BenchmarkQueryIdentity(
                query_id=case.query.query_id,
                split=case.split,
                source_dataset=case.query.source_dataset,
                query_tags=case.query.query_tags,
            )
            for case in cases
        ),
        created_at_utc="2026-08-19T00:00:00+00:00",
    )
    return protocol, manifest, cases


class _ProfileExecutor:
    def __init__(self, relevant_by_query: dict[str, str], manifest: ProfileRunManifest) -> None:
        self._relevant_by_query = relevant_by_query
        self._manifest = manifest
        self.calls = 0

    async def prepare_trial(self, invocation: ProfileInvocation) -> None:
        del invocation

    async def execute(self, invocation: ProfileInvocation) -> ProfileTrialObservation:
        self.calls += 1
        plan_id = invocation.plan_features.plan_id
        latency = {"P0": 10.0, "P2": 20.0, "P4": 30.0}[plan_id]
        features = QueryFeatures(
            token_count=7,
            entity_count=2,
            entity_density=2 / 7,
            relation_signal=0.5,
            multi_hop_signal=0.0,
            comparison_signal=0.0,
            aggregation_signal=0.0,
            global_signal=0.0,
            final_top_k=10,
        )
        branches: tuple[BranchTrialRecord, ...]
        status = TrialStatus.COMPLETE
        fallback = False
        if plan_id == "P0":
            branches = (
                BranchTrialRecord(
                    branch=BranchKind.VECTOR,
                    status=BranchStatus.SUCCEEDED,
                    latency_ms=latency,
                ),
            )
        elif plan_id == "P2":
            branches = (
                BranchTrialRecord(
                    branch=BranchKind.GRAPH,
                    status=BranchStatus.SUCCEEDED,
                    latency_ms=latency,
                ),
            )
        else:
            status = TrialStatus.PARTIAL
            fallback = True
            branches = (
                BranchTrialRecord(
                    branch=BranchKind.VECTOR,
                    status=BranchStatus.SUCCEEDED,
                    latency_ms=latency,
                ),
                BranchTrialRecord(
                    branch=BranchKind.GRAPH,
                    status=BranchStatus.FAILED,
                    latency_ms=latency,
                    error_code=ErrorCode.DEPENDENCY_UNAVAILABLE,
                ),
            )
        return ProfileTrialObservation(
            status=status,
            ranked_chunk_ids=(self._relevant_by_query[invocation.query_id],),
            selected_plan_id=plan_id,
            query_features=features,
            execution_latency_ms=latency,
            total_latency_ms=latency + 2.0,
            vector_latency_ms=latency if invocation.plan_features.vector_enabled else None,
            graph_latency_ms=latency if invocation.plan_features.graph_enabled else None,
            branch_results=branches,
            fallback=fallback,
            scheduler_trace_present=True,
            scheduler_runtime_semantics_version="v1",
            trace_config_version=self._manifest.plan_catalog_sha256,
            trace_model_version=(
                f"{self._manifest.embedding_model_revision}:{self._manifest.extractor_version}"
            ),
        )


@pytest.mark.asyncio
async def test_three_query_three_plan_profile_is_exact_and_byte_stable(
    tmp_path: Path,
    profile_fixture: tuple[object, object, tuple[BenchmarkCase, ...]],
) -> None:
    protocol, manifest, cases = profile_fixture
    assert hasattr(protocol, "plan_features")
    assert isinstance(manifest, ProfileRunManifest)
    repository = ProfileRepository(
        tmp_path,
        protocol=protocol,  # type: ignore[arg-type]
        environment=_environment(),
        manifest=manifest,
    )
    executor = _ProfileExecutor(
        {case.query.query_id: next(iter(case.relevance)) for case in cases},
        manifest,
    )
    summary = await PlanProfiler(
        protocol=protocol,  # type: ignore[arg-type]
        run_manifest=manifest,
        cases=cases,
        executor=executor,
        repository=repository,
    ).run()

    assert summary.complete
    assert summary.total_rows == 3 * 3 * 1 * 13
    records = repository.load_records()
    assert len({item.trial_id for item in records}) == len(records)
    assert all(
        item.final_engine_path == "BaselineSearchEngine.benchmark_plan_search" for item in records
    )
    matrix = build_training_matrix(records, manifest=manifest)
    assert len(matrix) == 3 * 3
    p4 = next(item for item in matrix if item.plan_id == "P4")
    assert p4.partial_trial_count == 10
    assert p4.full_result_recall_at_10 is None
    assert p4.partial_result_recall_at_10 is not None
    assert p4.partial_result_recall_at_10 > 0.0
    assert p4.fallback_rate == 1.0
    assert p4.usable_for_model_training is True

    write_training_matrix_artifacts(repository.run_dir, records=records, matrix=matrix)
    first = {
        name: (repository.run_dir / name).read_bytes()
        for name in ("training_matrix.jsonl", "training_matrix.csv", "checksums.json")
    }
    write_training_matrix_artifacts(repository.run_dir, records=records, matrix=matrix)
    assert first == {name: (repository.run_dir / name).read_bytes() for name in first}


@pytest.mark.asyncio
async def test_missing_trials_and_every_identity_mismatch_are_rejected(
    tmp_path: Path,
    profile_fixture: tuple[object, object, tuple[BenchmarkCase, ...]],
) -> None:
    protocol, manifest, cases = profile_fixture
    assert isinstance(manifest, ProfileRunManifest)
    repository = ProfileRepository(
        tmp_path,
        protocol=protocol,  # type: ignore[arg-type]
        environment=_environment(),
        manifest=manifest,
    )
    executor = _ProfileExecutor(
        {case.query.query_id: next(iter(case.relevance)) for case in cases},
        manifest,
    )
    await PlanProfiler(
        protocol=protocol,  # type: ignore[arg-type]
        run_manifest=manifest,
        cases=cases,
        executor=executor,
        repository=repository,
    ).run()
    records = repository.load_records()
    with pytest.raises(RawEvidenceError, match="incomplete"):
        build_training_matrix(records[:-1], manifest=manifest)
    mismatch_updates = (
        {"plan_catalog_sha256": "f" * 64, "trace_config_version": "f" * 64},
        {"corpus_version": "wrong-corpus"},
        {"environment_manifest_sha256": "e" * 64},
    )
    for update in mismatch_updates:
        mismatched = (records[0].model_copy(update=update), *records[1:])
        with pytest.raises((RawEvidenceError, ValueError), match="mismatch|version"):
            build_training_matrix(mismatched, manifest=manifest)

    measured_index = next(
        index for index, record in enumerate(records) if record.trial_phase.value == "measured"
    )
    runtime_drift = records[measured_index].model_copy(
        update={
            "scheduler_runtime_semantics_version": "v2",
            "execution_latency_label_valid": False,
            "invalid_exclusion_reason": "runtime_semantics_trace_mismatch",
        }
    )
    drifted = (*records[:measured_index], runtime_drift, *records[measured_index + 1 :])
    matrix = build_training_matrix(drifted, manifest=manifest)
    affected = next(
        item
        for item in matrix
        if item.query_id == runtime_drift.query_id
        and item.plan_id == runtime_drift.plan_id
        and item.latency_budget_ms == runtime_drift.latency_budget_ms
    )
    assert affected.usable_for_model_training is False
    assert affected.invalid_exclusion_reasons == ("runtime_semantics_trace_mismatch",)

    recovery_query_id = records[0].query_id
    recovery_features = next(
        record.query_features
        for record in records
        if record.query_id == recovery_query_id and record.query_features is not None
    )
    assert recovery_features is not None
    featureless = tuple(
        record.model_copy(
            update={
                "query_features": None,
                "quality_label_valid": False,
                "execution_latency_label_valid": False,
                "invalid_exclusion_reason": "query_features_missing",
            }
        )
        if record.query_id == recovery_query_id
        else record
        for record in records
    )
    with pytest.raises(RawEvidenceError, match="no profiler feature"):
        build_training_matrix(featureless, manifest=manifest)
    recovered = build_training_matrix(
        featureless,
        manifest=manifest,
        fallback_query_features={recovery_query_id: recovery_features},
    )
    recovered_rows = tuple(item for item in recovered if item.query_id == recovery_query_id)
    assert recovered_rows
    assert all(item.query_features == recovery_features for item in recovered_rows)
    assert all(item.usable_for_model_training is False for item in recovered_rows)


def test_profiler_refuses_held_out_test_split(
    tmp_path: Path,
    profile_fixture: tuple[object, object, tuple[BenchmarkCase, ...]],
) -> None:
    protocol, _, cases = profile_fixture
    test_case = BenchmarkCase(
        query=cases[0].query,
        split=SplitName.TEST,
        relevance=cases[0].relevance,
    )
    manifest = create_profile_run_manifest(
        run_id="stage10-test-refusal",
        protocol=protocol,  # type: ignore[arg-type]
        environment=_environment(),
        query_identities=(
            BenchmarkQueryIdentity(
                query_id=test_case.query.query_id,
                split=SplitName.TEST,
                source_dataset=test_case.query.source_dataset,
                query_tags=test_case.query.query_tags,
            ),
        ),
        created_at_utc="2026-08-19T00:00:00+00:00",
    )
    repository = ProfileRepository(
        tmp_path,
        protocol=protocol,  # type: ignore[arg-type]
        environment=_environment(),
        manifest=manifest,
    )
    with pytest.raises(ValueError, match="held-out test"):
        PlanProfiler(
            protocol=protocol,  # type: ignore[arg-type]
            run_manifest=manifest,
            cases=(test_case,),
            executor=_ProfileExecutor({}, manifest),
            repository=repository,
        )


@pytest.mark.asyncio
async def test_interrupted_profile_resumes_without_duplicate_trials(
    tmp_path: Path,
    profile_fixture: tuple[object, object, tuple[BenchmarkCase, ...]],
) -> None:
    protocol, source_manifest, cases = profile_fixture
    assert isinstance(source_manifest, ProfileRunManifest)
    manifest = source_manifest.model_copy(update={"run_id": "stage10-resume"})
    repository = ProfileRepository(
        tmp_path,
        protocol=protocol,  # type: ignore[arg-type]
        environment=_environment(),
        manifest=manifest,
    )
    delegate = _ProfileExecutor(
        {case.query.query_id: next(iter(case.relevance)) for case in cases},
        manifest,
    )

    class _Interrupted:
        calls = 0

        async def prepare_trial(self, invocation: ProfileInvocation) -> None:
            await delegate.prepare_trial(invocation)

        async def execute(self, invocation: ProfileInvocation) -> ProfileTrialObservation:
            self.calls += 1
            if self.calls > 7:
                raise asyncio.CancelledError
            return await delegate.execute(invocation)

    with pytest.raises(asyncio.CancelledError):
        await PlanProfiler(
            protocol=protocol,  # type: ignore[arg-type]
            run_manifest=manifest,
            cases=cases,
            executor=_Interrupted(),
            repository=repository,
        ).run()
    assert len(repository.load_records()) == 7
    summary = await PlanProfiler(
        protocol=protocol,  # type: ignore[arg-type]
        run_manifest=manifest,
        cases=cases,
        executor=delegate,
        repository=repository,
    ).run()
    assert summary.complete
    assert summary.preexisting_rows == 7
