"""End-to-end tests against a LIVE cluster (run inside slurmctld).

    make test-doctor          # from the repo root on the docker host

Submits every script in tests/failing_jobs/, waits for terminal states,
runs `slurm-doctor suggest` on each, and asserts the diagnosis category and
proposed fix kinds. Marked `e2e`; plain `pytest` skips these.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

FAILING = Path(__file__).parent / "failing_jobs"
JOBS_DIR = Path(os.environ.get("SLURM_DOCTOR_E2E_DIR", "/data/jobs"))

# script -> (expected category, acceptable top fix kinds, acceptable rule ids)
CASES = {
    "oom.sh": ("memory", {"bump_memory"}),
    "timeout.sh": ("time", {"bump_time"}),
    "bad_module.sh": ("environment", {"prepend_modules"}),
    "missing_exec.sh": ("exec", {"fix_path", "add_set_eux"}),
    "mpi_bad_launcher.sh": ("mpi", {"swap_mpi_launcher"}),
    "segfault.sh": ("application", {None, "add_set_eux"}),
    "cuda_missing.sh": ("gpu", {"request_constraint"}),
    "python_import.sh": ("application", {None, "add_set_eux"}),
    "disk_full.sh": ("disk", {None, "add_set_eux"}),
}

TERMINAL = ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL",
            "DEADLINE", "CANCELLED", "COMPLETED")


def _run(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, timeout=120, **kw)


@pytest.fixture(scope="module")
def submitted_jobs():
    if shutil.which("sbatch") is None:
        pytest.skip("no sbatch here - e2e tests run inside slurmctld")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, str] = {}
    for script in sorted(CASES):
        dst = JOBS_DIR / script
        shutil.copy(FAILING / script, dst)
        r = _run(["sbatch", str(dst)], cwd=JOBS_DIR)
        m = re.search(r"Submitted batch job (\d+)", r.stdout)
        assert m, f"sbatch {script} failed: {r.stderr}"
        jobs[script] = m.group(1)

    deadline = time.monotonic() + 360  # timeout.sh needs ~2 min to be killed
    pending = dict(jobs)
    while pending and time.monotonic() < deadline:
        time.sleep(5)
        for script, jobid in list(pending.items()):
            r = _run(["sacct", "-j", jobid, "-X", "--noheader", "--parsable2",
                      "--format=State,End"])
            line = r.stdout.strip().splitlines()
            if not line:
                continue
            state, end = (line[0].split("|") + [""])[:2]
            if end not in ("", "Unknown") and state.split()[0].split("+")[0] in TERMINAL:
                pending.pop(script)
    assert not pending, f"jobs never finished: {pending}"
    return jobs


@pytest.mark.parametrize("script", sorted(CASES))
def test_diagnosis_end_to_end(script, submitted_jobs, tmp_path_factory):
    jobid = submitted_jobs[script]
    report_dir = tmp_path_factory.mktemp("reports")
    r = _run(["slurm-doctor", "--report-dir", str(report_dir),
              "suggest", jobid, "--refresh"])
    assert r.returncode == 0, f"suggest failed: {r.stderr}\n{r.stdout}"

    data = json.loads((report_dir / jobid / "report.json").read_text())
    diag = data["diagnosis"]
    want_cat, want_kinds = CASES[script]
    assert diag["category"] == want_cat, (
        f"{script}: category {diag['category']!r} != {want_cat!r} "
        f"(rules: {diag['rule_ids']})"
    )
    top_kind = diag["fixes"][0]["kind"] if diag["fixes"] else None
    assert top_kind in want_kinds, (
        f"{script}: top fix {top_kind!r} not in {want_kinds!r}"
    )
    # the quality bar: every diagnosis must cite evidence, and the rule
    # engine must have resolved it without the LLM
    assert diag["evidence"], f"{script}: no evidence cited"
    assert diag["used_llm"] is False
    # markdown twin exists and leads with a one-line TL;DR
    md = (report_dir / jobid / "report.md").read_text()
    assert md.splitlines()[0].startswith("# slurm-doctor report")


def test_heal_oom_end_to_end(submitted_jobs, tmp_path_factory):
    """heal() on the OOM job must resubmit a patched script that survives."""
    jobid = submitted_jobs["oom.sh"]
    report_dir = tmp_path_factory.mktemp("heal-reports")
    r = _run(["slurm-doctor", "--report-dir", str(report_dir),
              "heal", jobid, "--refresh"])
    assert r.returncode == 0, f"heal failed: {r.stderr}\n{r.stdout}"
    m = re.search(r"resubmitted as job (\d+)", r.stdout)
    assert m, r.stdout
    new_id = m.group(1)

    deadline = time.monotonic() + 300
    state = ""
    while time.monotonic() < deadline:
        rr = _run(["sacct", "-j", new_id, "-X", "--noheader", "--parsable2",
                   "--format=State,End"])
        line = rr.stdout.strip().splitlines()
        if line:
            state, end = (line[0].split("|") + [""])[:2]
            if end not in ("", "Unknown"):
                break
        time.sleep(5)
    assert state.startswith("COMPLETED"), f"healed job {new_id} ended {state}"

    # the resubmission carries the audit comment
    rr = _run(["sacct", "-j", new_id, "-X", "--noheader", "--parsable2",
               "--format=Comment"])
    assert f"slurm-doctor:fix=bump_memory:parent={jobid}" in rr.stdout
