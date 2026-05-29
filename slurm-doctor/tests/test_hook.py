"""Tests for the jobcomp_hook.sh shell hook.

We invoke the real hook script with a stub `python3` so we can assert its
filtering + backgrounding behaviour without a live SLURM. The stub records
the args it was called with into a marker file.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "slurm_doctor" / "hooks" / "jobcomp_hook.sh"


def _run_hook(tmp_path: Path, env_extra: dict[str, str]) -> tuple[int, Path, float]:
    """Run the hook with a stub python; return (rc, marker_path, elapsed_s)."""
    sd_home = tmp_path / "home"
    sd_home.mkdir()
    marker = tmp_path / "invoked.txt"

    # Stub python: append all args to the marker file and exit 0.
    stub = tmp_path / "py-stub.sh"
    stub.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> "{marker}"\n'
    )
    stub.chmod(0o755)

    env = dict(os.environ)
    env.update({
        "SLURM_DOCTOR_HOME": str(sd_home),
        "SLURM_DOCTOR_PYTHON": str(stub),
        "SLURM_DOCTOR_REPORTS": str(tmp_path / "reports"),
        "SLURM_DOCTOR_CACHE": str(tmp_path / "cache"),
        "SLURM_DOCTOR_HOOK_LOG": str(tmp_path / "hook.log"),
    })
    env.update(env_extra)

    start = time.monotonic()
    proc = subprocess.run(["bash", str(HOOK)], env=env, timeout=10)
    elapsed = time.monotonic() - start
    return proc.returncode, marker, elapsed


def test_hook_acts_on_failure_state(tmp_path):
    rc, marker, elapsed = _run_hook(tmp_path, {"JOBID": "42", "JOBSTATE": "FAILED",
                                               "EXITCODE": "1:0"})
    assert rc == 0
    # the hook returns immediately; the stub runs detached — wait for it
    for _ in range(50):
        if marker.exists():
            break
        time.sleep(0.05)
    assert marker.exists(), "stub python was never invoked for a FAILED job"
    args = marker.read_text()
    assert "suggest 42" in args
    assert "--hook" in args
    assert "--reports-dir" in args


@pytest.mark.parametrize("state", ["COMPLETED", "RUNNING", "CANCELLED", ""])
def test_hook_ignores_non_failure_states(tmp_path, state):
    rc, marker, _elapsed = _run_hook(tmp_path, {"JOBID": "7", "JOBSTATE": state})
    assert rc == 0
    time.sleep(0.3)  # give any errant background job time to (not) fire
    assert not marker.exists(), f"hook should not act on JOBSTATE={state!r}"


def test_hook_returns_fast(tmp_path):
    # Foreground path must be trivial; real work is detached.
    _rc, _marker, elapsed = _run_hook(tmp_path, {"JOBID": "1", "JOBSTATE": "TIMEOUT"})
    assert elapsed < 1.0, f"hook took {elapsed:.2f}s; must return promptly"


def test_hook_noops_without_jobid(tmp_path):
    rc, marker, _ = _run_hook(tmp_path, {"JOBSTATE": "FAILED"})
    assert rc == 0
    time.sleep(0.2)
    assert not marker.exists()


def test_all_failure_states_are_covered():
    """The hook's failure set must match the states the parser treats as
    failures (so we never silently skip a state the engine can diagnose)."""
    hook_text = HOOK.read_text()
    for state in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                  "BOOT_FAIL", "DEADLINE"):
        assert state in hook_text, f"hook missing failure state {state}"
