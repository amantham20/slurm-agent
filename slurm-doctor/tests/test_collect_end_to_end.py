"""End-to-end Collector test with a mocked shell-out runner.

Runs on the host with no SLURM tools installed — proves the runner abstraction
buys us testability and that the orchestration glues the parsers together.
"""
from __future__ import annotations

import json
from pathlib import Path

from conftest import FIXTURES, read_fixture
from slurm_doctor._shell import ShellResult
from slurm_doctor.collect import Collector


def _make_runner(sacct_b_stdout: str, command_path: Path):
    """Return a fake runner closure with canned responses keyed off argv."""
    scontrol_text = read_fixture("scontrol_show_job_failed.txt").replace(
        "/data/sd_demo.sh", str(command_path)
    )

    def fake(argv, **kw):
        head = argv[:3]
        if argv[:2] == ["sacct", "-j"]:
            return ShellResult(
                argv=list(argv),
                returncode=0,
                stdout=read_fixture("sacct_failed_missing_exec.txt"),
                stderr="",
            )
        if argv[:2] == ["sacct", "-B"]:
            return ShellResult(argv=list(argv), returncode=0, stdout=sacct_b_stdout, stderr="")
        if head == ["scontrol", "show", "job"]:
            return ShellResult(argv=list(argv), returncode=0, stdout=scontrol_text, stderr="")
        if head == ["scontrol", "show", "node"]:
            return ShellResult(
                argv=list(argv),
                returncode=0,
                stdout=read_fixture("scontrol_show_node_c1.txt"),
                stderr="",
            )
        return ShellResult(argv=list(argv), returncode=127, stdout="", stderr="missing", missing=True)

    return fake


def test_collector_assembles_full_bundle_with_sacctB_fallback(tmp_path):
    # The cluster under test doesn't enable AccountingStoreFlags=job_script,
    # so sacct -B returns "NONE". The collector must fall back to reading
    # the Command= path off scontrol.
    cmd_path = tmp_path / "demo.sh"
    cmd_path.write_text("#!/bin/bash\necho hi from fixture\n")

    slurmd_log = tmp_path / "slurmd_c1.log"
    slurmd_log.write_text(read_fixture("slurmd_log_jid2.txt"))

    fake = _make_runner(
        sacct_b_stdout=f"Batch Script for 2\n{'-'*80}\nNONE\n",
        command_path=cmd_path,
    )

    cache = tmp_path / "cache"
    c = Collector(
        "2",
        cache_root=cache,
        runner=fake,
        slurmd_log_candidates=[str(tmp_path / "slurmd_{node}.log")],
    )
    bundle = c.collect()

    # Headline accounting
    assert bundle.state == "FAILED"
    assert bundle.exit_code == "127:0"
    assert bundle.derived_exit_code == "0:0"
    assert bundle.nodelist == "c1"
    assert bundle.req_mem == "200M"
    assert bundle.workdir == "/data"

    # Submit script via fallback (sacct -B was "NONE")
    assert bundle.submit_script_path is not None
    assert "hi from fixture" in Path(bundle.submit_script_path).read_text()

    # Per-node artifacts written
    assert "c1" in bundle.nodes
    assert bundle.nodes["c1"]["slurmd_log_path"]
    log_lines = Path(bundle.nodes["c1"]["slurmd_log_path"]).read_text()
    assert "JobId=2" in log_lines
    assert "exit code 127" in log_lines

    # Optional collectors skipped cleanly
    assert bundle.dmesg_path is None
    assert bundle.gpu == {}
    assert bundle.errors == []

    # bundle.json index written and self-consistent
    idx = json.loads((cache / "2" / "bundle.json").read_text())
    assert idx["jobid"] == "2"
    assert idx["schema_version"] == 1
    assert idx["state"] == "FAILED"


def test_collector_uses_sacctB_primary_when_script_present(tmp_path):
    # When AccountingStoreFlags=job_script IS enabled, sacct -B returns content;
    # the primary path should strip the 2-line header and write that.
    script_body = "#!/bin/bash\n#SBATCH --job-name=x\necho real script\n"
    sacct_b = f"Batch Script for 2\n{'-'*80}\n{script_body}"

    # Command path doesn't exist on disk — primary path must NOT need it.
    fake = _make_runner(sacct_b_stdout=sacct_b, command_path=Path("/nonexistent/x.sh"))

    cache = tmp_path / "cache"
    c = Collector(
        "2",
        cache_root=cache,
        runner=fake,
        slurmd_log_candidates=[str(tmp_path / "missing_{node}.log")],
    )
    bundle = c.collect()

    assert bundle.submit_script_path is not None
    written = Path(bundle.submit_script_path).read_text()
    assert written == script_body  # header stripped
    assert "Batch Script for 2" not in written


def test_collector_rejects_unsafe_jobid(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        Collector("2; rm -rf /", cache_root=tmp_path)


def test_collector_idempotent_overwrites_cache(tmp_path):
    cmd = tmp_path / "s.sh"
    cmd.write_text("#!/bin/bash\necho ok\n")
    fake = _make_runner(sacct_b_stdout="", command_path=cmd)
    c1 = Collector("2", cache_root=tmp_path / "cache", runner=fake,
                   slurmd_log_candidates=[str(tmp_path / "x_{node}.log")])
    b1 = c1.collect()
    sz1 = Path(b1.cache_dir).stat().st_mtime
    # Re-run should be safe — same artifacts, no exceptions.
    b2 = c1.collect()
    assert b2.state == b1.state == "FAILED"
    assert b2.errors == []
