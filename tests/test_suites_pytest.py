"""
Pytest wrappers for the script-style suites.

Historically only ``test_smoke.py`` was pytest-native, so ``pytest tests/``
reported a handful of tests while the substantive suites (``test_task3.py``,
``test_system.py``) ran only as standalone ``python tests/X.py`` scripts in CI.
That made ``pytest`` under-report real coverage.

These wrappers invoke the CPU-safe, checkpoint-free suites as subprocesses so
they are collected and reported by ``pytest`` like any other test, while the
underlying scripts remain runnable directly (and in CI) exactly as before. The
live-server phase 3 suite and the LFS-checkpoint RAG probes are intentionally
left to their direct CI invocations to keep ``pytest tests/`` fast and
hermetic.
"""

import os
import subprocess
import sys

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)


def _run(*args):
    """Run a suite as a subprocess under the tiny CPU config; assert exit 0."""
    env = dict(os.environ)
    env["TINY_MODEL"] = "1"
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"{' '.join(args)} exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}"
        )


def test_structural_suite():
    """tests/test_task3.py — embedding / RMSNorm / output-head structural checks."""
    _run(os.path.join("tests", "test_task3.py"))


def test_system_suite_phase1_unit():
    """tests/test_system.py --phase 1 — unit checks (41)."""
    _run(os.path.join("tests", "test_system.py"), "--phase", "1")


def test_system_suite_phase2_integration():
    """tests/test_system.py --phase 2 — integration checks (20)."""
    _run(os.path.join("tests", "test_system.py"), "--phase", "2")
