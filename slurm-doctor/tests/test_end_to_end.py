"""End-to-end tests against a LIVE Slurm cluster.

These submit each deliberately-broken job, wait for it to finish, run the full
collect -> diagnose pipeline, and assert the diagnosed category + a proposed
fix kind. They require sbatch/sacct on PATH, so they're skipped on a plain host
and are meant to run inside slurmctld via `make test-doctor`.

Confirmed on Slurm 25.11 in slurm-docker-cluster: all 9 categories are resolved
by the rule engine with ZERO LLM calls.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from slurm_doctor.collect import collect
from slurm_doctor.diagnose import diagnose
from slurm_doctor.parse import load_rules

pytestmark = pytest.mark.skipif(
    shutil.which("sbatch") is None or shutil.which("sacct") is None,
    reason="needs a live Slurm (sbatch/sacct) — run via `make test-doctor`",
)

JOBS_DIR = Path(__file__).resolve().parent / "failing_jobs"
TERMINAL = {"FAILED", "COMPLETED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "CANCELLED", "BOOT_FAIL", "DEADLINE"}

# script stem -> (acceptable categories, at least one of these fix kinds)
# An empty fix set means "no automated fix is expected" (e.g. disk full).
CASES = [
    ("oom",              {"python_memory", "memory_heuristic", "memory_slurm"}, {"bump_memory"}),
    ("timeout",          {"time"},            {"bump_time"}),
    ("bad_module",       {"modules"},         {"prepend_modules"}),
    ("missing_exec",     {"environment"},     {"fix_path", "add_set_eux"}),
    ("mpi_bad_launcher", {"mpi"},             {"swap_mpi_launcher"}),
    ("segfault",         {"native_crash"},    {"add_set_eux"}),
    ("cuda_missing",     {"gpu_driver"},      {"request_constraint"}),
    ("python_import",    {"python_runtime"},  {"prepend_modules"}),
    ("disk_full",        {"disk"},            set()),
]


def _submit(script: Path) -> str:
    out = subprocess.run(
        ["sbatch", "--parsable", str(script)],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout.strip()
    return out.split(";")[0]


def _wait_terminal(jobid: str, timeout_s: int = 90) -> str:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        r = subprocess.run(
            ["sacct", "-j", jobid, "-n", "-X", "-o", "State"],
            capture_output=True, text=True, timeout=15,
        )
        last = (r.stdout.splitlines() or [""])[0].strip().split()[0] if r.stdout.strip() else ""
        if last in TERMINAL:
            return last
        time.sleep(2)
    return last


@pytest.fixture(scope="module")
def rules():
    return load_rules()


@pytest.mark.parametrize("stem,categories,fix_kinds", CASES, ids=[c[0] for c in CASES])
def test_failing_job_is_diagnosed(stem, categories, fix_kinds, rules, tmp_path_factory):
    script = JOBS_DIR / f"{stem}.sh"
    assert script.exists(), f"missing failing-job script: {script}"

    jobid = _submit(script)
    state = _wait_terminal(jobid)
    assert state in TERMINAL, f"{stem} (job {jobid}) never reached a terminal state (last={state!r})"
    assert state != "COMPLETED", f"{stem} (job {jobid}) unexpectedly succeeded"

    cache = tmp_path_factory.mktemp(f"cache_{stem}")
    bundle = collect(jobid, cache_root=str(cache))
    diag = diagnose(bundle, rules)

    seen = {diag.state_category} | {h["category"] for h in diag.rule_hits}
    assert categories & seen, (
        f"{stem} (job {jobid}, SLURM={state}): expected one of {categories}, "
        f"got categories {seen}; tldr={diag.tldr!r}"
    )

    proposed = {f.fix_kind for f in diag.proposed_fixes}
    if fix_kinds:
        assert fix_kinds & proposed, (
            f"{stem} (job {jobid}): expected a fix in {fix_kinds}, got {proposed}"
        )

    # Quality bar: rule engine resolves these without the LLM.
    assert diag.used_llm is False

    # Every diagnosis must cite evidence (no diagnosis without a citation),
    # except pure state-machine verdicts (timeout) which cite the state line.
    if diag.rule_hits:
        assert any(h["evidence"] for h in diag.rule_hits), f"{stem}: no evidence cited"

# NOTE: the parametrized cases above collectively prove the quality bar —
# all 9 categories resolve via the rule engine with used_llm is False, i.e.
# 100% (> 80%) with zero LLM calls. We don't re-submit them a second time
# just to recompute that fraction (the timeout job alone costs ~30s/run).
