"""Unit tests for collect.py against captured fixtures + mocked commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import slurm_doctor.collect as collect_mod
from slurm_doctor.collect import (
    Bundle,
    collect,
    parse_parsable2,
    parse_scontrol_kv,
)
from slurm_doctor.util import CmdResult

from conftest import FIXTURES, load_bundle


def test_parse_parsable2_fixture():
    raw = (FIXTURES / "job2" / "sacct.parsable2.txt").read_text()
    recs = parse_parsable2(raw)
    assert len(recs) == 2
    parent, batch = recs
    assert parent["JobID"] == "2" and parent["State"] == "FAILED"
    assert batch["JobID"] == "2.batch" and batch["MaxRSS"] == "310500K"


def test_parse_scontrol_kv_multiline_values():
    text = (
        "JobId=5 JobName=missing_exec\n"
        "   Command=/data/jobs/missing_exec.sh --flag value\n"
        "   WorkDir=/data/jobs\n"
        "   StdErr=/data/jobs/missing_exec-5.out\n"
        "   Partition=cpu AllocNode:Sid=slurmctld:1\n"
    )
    kv = parse_scontrol_kv(text)
    assert kv["JobId"] == "5"
    assert kv["Command"] == "/data/jobs/missing_exec.sh --flag value"
    assert kv["StdErr"] == "/data/jobs/missing_exec-5.out"
    assert kv["Partition"] == "cpu"


def test_bundle_accessors_from_fixture(oom_bundle: Bundle):
    assert oom_bundle.state == "FAILED"
    assert oom_bundle.is_terminal
    assert oom_bundle.parent["JobName"] == "oom"
    assert oom_bundle.steps[0]["JobID"] == "2.batch"
    assert oom_bundle.max_rss_bytes() == 310500 * 1024
    assert "exceeded memory limit" in (oom_bundle.stderr_text or "")
    srcs = oom_bundle.evidence_sources()
    assert {"stderr", "script", "sacct"} <= set(srcs)


def test_collect_uses_cache_without_commands(oom_bundle, cfg, monkeypatch):
    """A cached terminal job must not shell out at all."""
    cache = cfg.job_cache("2")
    cache.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copytree(oom_bundle.dir, cache)

    def boom(argv, **kw):
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(collect_mod, "run", boom)
    b = collect("2", cfg)
    assert b.state == "FAILED"


def test_collect_full_run_mocked(cfg, monkeypatch, tmp_path):
    """Drive a full collection from canned command outputs."""
    sacct_out = (FIXTURES / "job2" / "sacct.parsable2.txt").read_text()
    script = (FIXTURES / "job2" / "batch_script.sh").read_text()
    stdout_file = tmp_path / "oom-2.out"
    stdout_file.write_text("allocated 300MB, holding...\nerror: exceeded memory limit\n")

    def fake_run(argv, **kw):
        cmd = " ".join(argv)
        if argv[0] == "sacct" and "--parsable2" in cmd and "--format=J" in cmd:
            return CmdResult(argv, 0, sacct_out, "", 0.0)
        if argv[0] == "sacct" and "-B" in argv:
            return CmdResult(argv, 0, f"Batch Script for 2\n{'-' * 20}\n{script}", "", 0.0)
        if argv[0] == "sacct" and "--env-vars" in cmd:
            return CmdResult(argv, 0, "PATH=/usr/bin\nMY_TOKEN=secret123\n", "", 0.0)
        if argv[:3] == ["scontrol", "show", "job"]:
            return CmdResult(
                argv, 0,
                f"JobId=2 JobName=oom\n   StdOut={stdout_file}\n   StdErr={stdout_file}\n"
                f"   WorkDir={tmp_path}\n   Command={tmp_path}/oom.sh\n",
                "", 0.0,
            )
        if argv[:3] == ["scontrol", "show", "node"]:
            return CmdResult(argv, 0, "NodeName=c1 State=IDLE\n", "", 0.0)
        if argv[:3] == ["scontrol", "show", "partition"]:
            return CmdResult(argv, 0, "PartitionName=cpu MaxTime=UNLIMITED\n", "", 0.0)
        if argv[0] == "dmesg":
            return CmdResult(argv, 1, "", "dmesg: permission denied", 0.0)
        return CmdResult(argv, 1, "", f"unhandled: {cmd}", 0.0)

    monkeypatch.setattr(collect_mod, "run", fake_run)
    monkeypatch.setattr(collect_mod, "SLURMD_LOG", str(tmp_path / "absent.log"))
    monkeypatch.setattr(collect_mod, "JOBCOMP_LOG", str(tmp_path / "absent2.log"))

    b = collect("2", cfg)
    assert b.parent["State"] == "FAILED"
    assert "exceeded memory limit" in b.stderr_text
    assert b.script and b.script.startswith("#!/bin/bash")
    # stored env must be redacted at rest
    env = (b.dir / "env.redacted.txt").read_text()
    assert "secret123" not in env and "PATH=/usr/bin" in env
    manifest = json.loads((b.dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["parent_summary"]["JobName"] == "oom"


def test_collect_rejects_garbage_jobid(cfg):
    with pytest.raises(ValueError):
        collect("; rm -rf /", cfg)


def test_io_paths_fall_back_to_script_directives(cfg):
    """Purged job: no scontrol, paths come from #SBATCH --output."""
    from slurm_doctor.collect import _io_paths

    script = "#!/bin/bash\n#SBATCH --output=/data/jobs/%x-%j.out\n"
    parent = {"WorkDir": "/data/jobs", "JobIDRaw": "7", "JobID": "7"}
    meta = {"jobid": "7", "jobname": "segfault", "user": "root", "node": "c1"}
    out, err = _io_paths({}, script, parent, meta)
    assert out == "/data/jobs/segfault-7.out"
    assert err == out  # stderr merges into stdout when -e is absent
