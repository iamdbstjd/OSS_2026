"""RAGPlan command-line interface."""

import typer

from ragplan import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True)


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


def main() -> None:
    """Run the command-line application."""
    app()
