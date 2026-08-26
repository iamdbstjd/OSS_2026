from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.e2e


def test_wheel_runs_smoke_qa_from_repository_independent_workdir() -> None:
    configured = os.environ.get("RAGPLAN_TEST_WHEEL", "").strip()
    if not configured:
        pytest.skip("set RAGPLAN_TEST_WHEEL to run clean-wheel E2E")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_clean_wheel.py"),
            "--wheel",
            configured,
            "--level",
            "smoke",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "passed"
    assert report["repository_independent_workdir"] is True
    assert report["held_out_test_accessed"] is False
