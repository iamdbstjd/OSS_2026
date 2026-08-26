#!/usr/bin/env python3
"""Install one wheel outside the repository and run its fixed Stage 13 QA command."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--level", choices=("smoke", "vector", "full"), default="smoke")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--api-url")
    parser.add_argument("--model-cache", type=Path, default=Path("models/minilm"))
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wheel = args.wheel.resolve(strict=True)
    model_cache = args.model_cache.resolve()
    if args.level == "full" and args.api_url is None:
        raise ValueError("clean-wheel full QA requires --api-url")
    with tempfile.TemporaryDirectory(prefix="ragplan-clean-wheel-") as directory:
        root = Path(directory)
        environment_dir = root / "venv"
        workdir = root / "empty-workdir"
        workdir.mkdir()
        venv.EnvBuilder(
            with_pip=True,
            clear=True,
            system_site_packages=False,
        ).create(environment_dir)
        python = environment_dir / "bin" / "python"
        executable = environment_dir / "bin" / "ragplan"
        install = [str(python), "-m", "pip", "install"]
        install.append(str(wheel))
        _run_checked(install, cwd=workdir)
        command = [str(executable), "qa", "--level", args.level]
        if args.level == "vector":
            command.extend(("--qdrant-url", args.qdrant_url, "--model-cache", str(model_cache)))
        if args.level == "full" and args.api_url is not None:
            command.extend(("--api-url", args.api_url))
        completed = _run_checked(command, cwd=workdir)
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("clean-wheel QA returned invalid JSON") from exc
        if (
            report.get("status") != "passed"
            or report.get("level") != args.level
            or report.get("held_out_test_accessed") is not False
        ):
            raise RuntimeError("clean-wheel QA did not satisfy its fixed contract")
        print(
            json.dumps(
                {
                    "schema_version": "clean_wheel_e2e_v1",
                    "status": "passed",
                    "level": args.level,
                    "wheel": wheel.name,
                    "repository_independent_workdir": True,
                    "held_out_test_accessed": False,
                    "qa_check_count": len(report["checks"]),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    return 0


def _run_checked(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"} and not key.startswith("RAGPLAN_STAGE")
    }
    environment["RAGPLAN_LOGGING__MODE"] = "redacted"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"clean-wheel subprocess failed: {Path(command[0]).name} exit={completed.returncode}"
        )
    return completed


if __name__ == "__main__":
    raise SystemExit(run())
