"""Stage 9 protocol loading and reproducible environment capture."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Final

import yaml  # type: ignore[import-untyped]

from ragplan.benchmark.contracts import canonical_sha256
from ragplan.benchmark.records import (
    BenchmarkProtocolConfig,
    BenchmarkQueryIdentity,
    BenchmarkRunManifest,
    EnvironmentManifest,
    benchmark_query_identities_sha256,
)

DEFAULT_BASELINE_CONFIG_PATH: Final = (
    Path(__file__).resolve().parents[3] / "benchmark" / "configs" / "baseline_v1.yaml"
)
DEFAULT_DB_TUNING_CONFIG_PATH: Final = (
    Path(__file__).resolve().parents[3] / "benchmark" / "configs" / "db_tuning_default_v1.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source_sha256(repository_root: Path) -> str:
    """Hash the executable package/config/build inputs behind the local API image."""

    roots = (repository_root / "src/ragplan", repository_root / "configs")
    paths = [repository_root / "Dockerfile", repository_root / "pyproject.toml"]
    for root in roots:
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".json", ".py", ".yaml", ".yml"}
        )
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(repository_root).as_posix()):
        relative = path.relative_to(repository_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def load_benchmark_protocol(path: Path | None = None) -> BenchmarkProtocolConfig:
    try:
        if path is not None:
            serialized = path.read_text(encoding="utf-8")
        elif DEFAULT_BASELINE_CONFIG_PATH.is_file():
            serialized = DEFAULT_BASELINE_CONFIG_PATH.read_text(encoding="utf-8")
        else:
            serialized = (
                files("ragplan")
                .joinpath("resources", "benchmark", "baseline_v1.yaml")
                .read_text(encoding="utf-8")
            )
        payload = yaml.safe_load(serialized)
        return BenchmarkProtocolConfig.model_validate(payload, strict=False)
    except Exception as exc:
        raise ValueError("benchmark protocol config is missing or invalid") from exc


def load_environment_manifest(path: Path) -> EnvironmentManifest:
    try:
        return EnvironmentManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("benchmark environment manifest is missing or invalid") from exc


def capture_environment_manifest(
    repository_root: Path,
    *,
    cpu_governor: str | None = None,
    container_resource_limits: str = "unbounded (recorded)",
    db_tuning_sha256: str | None = None,
    notes: str = "dedicated benchmark session; operator must confirm no competing workload",
) -> EnvironmentManifest:
    """Capture non-secret host/runtime evidence; never infer performance comparability silently."""

    repository_root = repository_root.resolve()
    compose_path = repository_root / "docker-compose.yml"
    lock_path = repository_root / "uv.lock"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = compose.get("services", {}) if isinstance(compose, dict) else {}

    def image(service: str) -> str:
        value = services.get(service, {}).get("image") if isinstance(services, dict) else None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Compose service {service!r} has no recorded image")
        return value

    observed_governor = cpu_governor or _read_cpu_governor()
    tuning_hash = db_tuning_sha256 or _default_db_tuning_sha256()
    if len(tuning_hash) != 64 or any(
        character not in "0123456789abcdef" for character in tuning_hash
    ):
        raise ValueError("DB tuning identity must be a lowercase SHA-256")
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        raise ValueError("logical CPU count is unavailable")
    return EnvironmentManifest(
        captured_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        os_name=platform.system() or "unknown",
        os_release=platform.release() or "unknown",
        machine=platform.machine() or "unknown",
        cpu_model=_read_cpu_model(),
        logical_cpu_count=cpu_count,
        cpu_governor=observed_governor,
        python_version=platform.python_version(),
        qdrant_image=image("qdrant"),
        neo4j_image=image("neo4j"),
        api_image=image("api"),
        container_resource_limits=container_resource_limits,
        runtime_source_sha256=runtime_source_sha256(repository_root),
        dependency_lock_sha256=file_sha256(lock_path),
        docker_compose_sha256=file_sha256(compose_path),
        db_tuning_sha256=tuning_hash,
        notes=notes,
    )


def verify_environment_manifest(repository_root: Path, expected: EnvironmentManifest) -> None:
    """Reject reuse after any observable host, lock, image, or governor drift."""

    observed = capture_environment_manifest(
        repository_root,
        container_resource_limits=expected.container_resource_limits,
        db_tuning_sha256=expected.db_tuning_sha256,
        notes=expected.notes,
    ).model_copy(update={"captured_at_utc": expected.captured_at_utc})
    if observed != expected:
        raise ValueError("current runtime differs from the recorded benchmark environment")


def create_run_manifest(
    *,
    run_id: str,
    protocol: BenchmarkProtocolConfig,
    environment: EnvironmentManifest,
    query_identities: Sequence[BenchmarkQueryIdentity],
    created_at_utc: str | None = None,
) -> BenchmarkRunManifest:
    if environment.concurrency != protocol.concurrency:
        raise ValueError("environment and benchmark protocol concurrency differ")
    timestamp = created_at_utc or datetime.now(UTC).isoformat(timespec="seconds")
    ordered_identities = tuple(sorted(query_identities, key=lambda item: item.query_id))
    ordered_query_ids = tuple(item.query_id for item in ordered_identities)
    identities_hash = benchmark_query_identities_sha256(ordered_identities)
    query_count = len(ordered_query_ids)
    expected_rows = (
        query_count
        * len(protocol.methods)
        * len(protocol.latency_budgets_ms)
        * protocol.trials_per_query_method_budget
    )
    return BenchmarkRunManifest(
        run_id=run_id,
        created_at_utc=timestamp,
        protocol_config_sha256=protocol.sha256,
        environment_manifest_sha256=environment.sha256,
        benchmark_manifest_sha256=protocol.benchmark_manifest_sha256,
        split_hash=protocol.split_hash,
        qrels_sha256=protocol.qrels_sha256,
        corpus_version=protocol.corpus_version,
        corpus_chunk_count=protocol.corpus_chunk_count,
        corpus_chunk_ids_sha256=protocol.corpus_chunk_ids_sha256,
        embedding_model_revision=protocol.embedding_model_revision,
        extractor_version=protocol.extractor_version,
        plan_catalog_sha256=protocol.plan_catalog_sha256,
        planner_config_sha256=protocol.planner_config_sha256,
        query_feature_config_sha256=protocol.query_feature_config_sha256,
        graph_tier_policy_sha256=protocol.graph_tier_policy_sha256,
        rule_runtime_config_version=protocol.rule_runtime_config_version,
        stage2_artifact_set_sha256=protocol.stage2_artifact_set_sha256,
        runtime_semantics_version=protocol.runtime_semantics_version,
        random_seed=protocol.random_seed,
        cold_runs=protocol.cold_runs,
        warmup_runs=protocol.warmup_runs,
        measured_runs=protocol.measured_runs,
        concurrency=protocol.concurrency,
        query_count=query_count,
        query_ids_sha256=canonical_sha256(list(ordered_query_ids)),
        query_identities_sha256=identities_hash,
        method_count=9,
        latency_budgets_ms=protocol.latency_budgets_ms,
        expected_raw_row_count=expected_rows,
    )


def _read_cpu_governor() -> str:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable (recorded)"
    return value or "unavailable (recorded)"


def _default_db_tuning_sha256() -> str:
    if DEFAULT_DB_TUNING_CONFIG_PATH.is_file():
        return file_sha256(DEFAULT_DB_TUNING_CONFIG_PATH)
    payload = (
        files("ragplan")
        .joinpath("resources", "benchmark", "db_tuning_default_v1.json")
        .read_bytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _read_cpu_model() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.casefold().startswith("model name") and ":" in line:
                    value = line.split(":", 1)[1].strip()
                    if value:
                        return value
        except OSError:
            pass
    return platform.processor() or "unknown"
