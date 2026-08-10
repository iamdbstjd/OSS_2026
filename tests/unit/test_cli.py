import pytest
from typer.testing import CliRunner

from ragplan import __version__
from ragplan.cli.app import app

pytestmark = pytest.mark.unit


def test_version_option_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout == f"{__version__}\n"
