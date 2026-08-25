"""RAGPlan command-line interface."""

import asyncio
import json
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, Never
from uuid import uuid4

import typer

from ragplan import __version__
from ragplan.backends.vector.qdrant import DEFAULT_COLLECTION_PREFIX
from ragplan.cli.services import (
    explain_plan,
    ingest_result_payload,
    json_line,
    quickstart_vector,
    search_configured_runtime,
    search_http_api,
    verify_installation,
)
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import PlannerMode, SearchRequest
from ragplan.ingestion.service import ingest_vector_corpus

app = typer.Typer(add_completion=False, no_args_is_help=True)
benchmark_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(
    benchmark_app,
    name="benchmark",
    help="Run frozen Stage 9 benchmark and Stage 10 profiler workflows.",
)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def cli(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_show_version,
        is_eager=True,
        help="Show the RAGPlan version and exit.",
    ),
) -> None:
    """RAGPlan retrieval optimizer commands."""


@app.command("demo-plan")
def demo_plan(
    query: str = typer.Option(..., "--query", help="Query text; output contains only its hash."),
    budget_ms: int = typer.Option(100, "--budget-ms", min=25, max=5000),
    top_k: int = typer.Option(10, "--top-k", min=1, max=50),
    entity_count: int = typer.Option(
        0,
        "--entity-count",
        min=0,
        help="Optional caller-supplied count; no NER model is loaded.",
    ),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Explain a Rule plan in under a minute without a model, DB, or retrieval."""

    _command(
        lambda: explain_plan(
            query=query,
            latency_budget_ms=budget_ms,
            top_k=top_k,
            entity_count=entity_count,
        ),
        request_id=f"demo-plan-{uuid4()}",
        pretty=pretty,
    )


@app.command("search")
def search(
    query: str = typer.Option(..., "--query", help="Query text; never written to trace output."),
    planner: PlannerMode = typer.Option(PlannerMode.RULE, "--planner"),
    top_k: int = typer.Option(10, "--top-k", min=1, max=50),
    budget_ms: int = typer.Option(200, "--budget-ms", min=25, max=5000),
    plan_id: str | None = typer.Option(None, "--plan-id"),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Optional API origin. Without it, the configured local runtime is used.",
    ),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Search through the same validated engine used by the REST API."""

    request_id = f"cli-search-{uuid4()}"
    try:
        request = SearchRequest(
            query=query,
            planner=planner,
            top_k=top_k,
            latency_budget_ms=budget_ms,
            plan_id=plan_id,
        )
    except ValueError as exc:
        del exc
        _fail(
            RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "search request is invalid",
                retryable=False,
            ),
            request_id,
        )
    if api_url is None:

        def operation() -> object:
            return _run(search_configured_runtime(request, request_id=request_id))
    else:

        def operation() -> object:
            return search_http_api(
                request,
                request_id=request_id,
                api_url=api_url,
            )

    _command(operation, request_id=request_id, pretty=pretty)


@app.command("ingest")
def ingest(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    corpus_version: str = typer.Option(..., "--corpus-version"),
    stage_manifest: Path = typer.Option(..., "--stage-manifest"),
    model_snapshot: Path | None = typer.Option(None, "--model-snapshot"),
    model_cache: Path = typer.Option(Path("models/minilm"), "--model-cache"),
    chunks_output: Path | None = typer.Option(None, "--chunks-output"),
    qdrant_url: str = typer.Option("http://127.0.0.1:6333", "--qdrant-url"),
    collection_prefix: str = typer.Option(DEFAULT_COLLECTION_PREFIX, "--collection-prefix"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Chunk, embed, and idempotently stage a strict corpus in Qdrant."""

    request_id = f"cli-ingest-{uuid4()}"
    _command(
        lambda: ingest_result_payload(
            _run(
                ingest_vector_corpus(
                    input_path=input_path,
                    corpus_version=corpus_version,
                    stage_manifest_path=stage_manifest,
                    model_snapshot=model_snapshot,
                    model_cache=model_cache,
                    chunks_output=chunks_output,
                    qdrant_url=qdrant_url,
                    collection_prefix=collection_prefix,
                )
            )
        ),
        request_id=request_id,
        pretty=pretty,
    )


@app.command("verify")
def verify(
    configured_runtime: bool = typer.Option(
        False,
        "--configured-runtime",
        help="Also construct and health-check the runtime selected by RAGPLAN_STAGE* variables.",
    ),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Verify packaged contracts, or additionally verify configured live dependencies."""

    _command(
        lambda: _run(verify_installation(configured_runtime=configured_runtime)),
        request_id=f"cli-verify-{uuid4()}",
        pretty=pretty,
    )


@app.command("quickstart-vector")
def quickstart_vector_command(
    query: str = typer.Option(
        "What did Ada Lovelace write about?",
        "--query",
        help="Sample query; only its hash appears in the returned trace.",
    ),
    input_path: Path = typer.Option(
        Path("examples/sample_corpus.json"),
        "--input",
        exists=True,
        dir_okay=False,
    ),
    model_cache: Path = typer.Option(Path("models/minilm"), "--model-cache"),
    stage_manifest: Path = typer.Option(
        Path("artifacts/quickstart-vector-stage.json"),
        "--stage-manifest",
    ),
    corpus_version: str = typer.Option("ragplan-quickstart-v2", "--corpus-version"),
    qdrant_url: str = typer.Option("http://127.0.0.1:6333", "--qdrant-url"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Download the pinned model, ingest the sample, and run one vector search."""

    _command(
        lambda: _run(
            quickstart_vector(
                query=query,
                input_path=input_path,
                model_cache=model_cache,
                stage_manifest_path=stage_manifest,
                corpus_version=corpus_version,
                qdrant_url=qdrant_url,
            )
        ),
        request_id=f"cli-quickstart-{uuid4()}",
        pretty=pretty,
    )


@benchmark_app.command("capture-environment")
def benchmark_capture_environment(
    output: Path = typer.Option(..., "--output"),
    container_resource_limits: str = typer.Option(..., "--container-resource-limits"),
    confirm_dedicated: bool = typer.Option(False, "--confirm-dedicated"),
) -> None:
    """Capture non-secret hardware/runtime evidence for one dedicated run."""

    arguments = [
        "capture-environment",
        "--output",
        str(output),
        "--container-resource-limits",
        container_resource_limits,
    ]
    if confirm_dedicated:
        arguments.append("--confirm-dedicated")
    raise typer.Exit(_run_benchmark(arguments))


@benchmark_app.command("run")
def benchmark_run(
    run_id: str = typer.Option(..., "--run-id"),
    environment_manifest: Path = typer.Option(..., "--environment-manifest"),
    config: Path | None = typer.Option(None, "--config"),
    output_root: Path | None = typer.Option(None, "--output-root"),
    confirm_dedicated: bool = typer.Option(False, "--confirm-dedicated"),
) -> None:
    """Resume the exact 480-query train/validation baseline matrix."""

    arguments = [
        "run",
        "--run-id",
        run_id,
        "--environment-manifest",
        str(environment_manifest),
    ]
    if config is not None:
        arguments.extend(("--config", str(config)))
    if output_root is not None:
        arguments.extend(("--output-root", str(output_root)))
    if confirm_dedicated:
        arguments.append("--confirm-dedicated")
    raise typer.Exit(_run_benchmark(arguments))


@benchmark_app.command("aggregate")
def benchmark_aggregate(
    run_id: str = typer.Option(..., "--run-id"),
    output_root: Path | None = typer.Option(None, "--output-root"),
) -> None:
    """Rebuild deterministic CSV/JSON/checksum artifacts from a complete raw run."""

    arguments = ["aggregate", "--run-id", run_id]
    if output_root is not None:
        arguments.extend(("--output-root", str(output_root)))
    raise typer.Exit(_run_benchmark(arguments))


@benchmark_app.command("profile")
def benchmark_profile(
    run_id: str = typer.Option(..., "--run-id"),
    environment_manifest: Path = typer.Option(..., "--environment-manifest"),
    baseline_config: Path | None = typer.Option(None, "--baseline-config"),
    output_root: Path | None = typer.Option(None, "--output-root"),
    confirm_dedicated: bool = typer.Option(False, "--confirm-dedicated"),
) -> None:
    """Resume the Stage 10 train/validation query-by-plan profile."""

    arguments = [
        "run",
        "--run-id",
        run_id,
        "--environment-manifest",
        str(environment_manifest),
    ]
    if baseline_config is not None:
        arguments.extend(("--baseline-config", str(baseline_config)))
    if output_root is not None:
        arguments.extend(("--output-root", str(output_root)))
    if confirm_dedicated:
        arguments.append("--confirm-dedicated")
    raise typer.Exit(_run_profiler(arguments))


@benchmark_app.command("profile-aggregate")
def benchmark_profile_aggregate(
    run_id: str = typer.Option(..., "--run-id"),
    output_root: Path | None = typer.Option(None, "--output-root"),
) -> None:
    """Rebuild the Stage 10 training matrix, Oracle labels, and checksums."""

    arguments = ["aggregate", "--run-id", run_id]
    if output_root is not None:
        arguments.extend(("--output-root", str(output_root)))
    raise typer.Exit(_run_profiler(arguments))


def _run_benchmark(arguments: list[str]) -> int:
    from ragplan.benchmark.command import run

    return run(arguments)


def _run_profiler(arguments: list[str]) -> int:
    from ragplan.benchmark.profile_command import run

    return run(arguments)


def _run[ResultT](awaitable: Coroutine[Any, Any, ResultT]) -> ResultT:
    return asyncio.run(awaitable)


def _command(operation: Callable[[], object], *, request_id: str, pretty: bool) -> None:
    try:
        result = operation()
    except RAGPlanError as exc:
        _fail(exc, request_id)
    except (OSError, TypeError, ValueError) as exc:
        del exc
        _fail(
            RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "command input is invalid",
                retryable=False,
            ),
            request_id,
        )
    payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    if pretty:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(json_line(payload))


def _fail(error: RAGPlanError, request_id: str) -> Never:
    typer.echo(json_line(error.response(request_id)), err=True)
    raise typer.Exit(1)


def main() -> None:
    """Run the command-line application."""
    app()
