"""RAGPlan command-line interface."""

from pathlib import Path

import typer

from ragplan import __version__

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


def main() -> None:
    """Run the command-line application."""
    app()
