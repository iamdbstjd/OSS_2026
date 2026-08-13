import pytest
from typer.testing import CliRunner

from ragplan import __version__
from ragplan.cli.app import app

pytestmark = pytest.mark.unit


def test_version_option_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout == f"{__version__}\n"


def test_benchmark_subcommands_are_exposed() -> None:
    result = CliRunner().invoke(app, ["benchmark", "--help"])

    assert result.exit_code == 0
    assert "capture-environment" in result.stdout
    assert "aggregate" in result.stdout
    assert "run" in result.stdout


def test_primary_benchmark_requires_dedicated_environment_confirmation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "run",
            "--run-id",
            "stage9-test",
            "--environment-manifest",
            str(tmp_path / "missing.json"),
        ],
    )

    assert result.exit_code == 1
    assert "--confirm-dedicated" in result.stderr
