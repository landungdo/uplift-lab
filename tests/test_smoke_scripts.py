"""
Smoke tests: every module's __main__ demo and the reproduce script must run
without error.

These exist because a demo can break (e.g. a class referenced before it is
defined) while the unit tests stay green — the unit tests import symbols
directly and never execute the __main__ blocks. Running the scripts as
subprocesses catches that class of error in CI.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "src/synthetic_data.py",
    "src/meta_learners.py",
    "src/evaluation.py",
    "src/policy.py",
    "src/experiment_design.py",
    "src/off_policy.py",
    "src/bootstrap.py",
    "src/ope_diagnostics.py",
    "src/cuped.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs(script):
    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"{script} failed:\n{result.stderr[-1500:]}"
    )
