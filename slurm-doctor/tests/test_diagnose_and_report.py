"""Tests for diagnose.py + report.py.

Covers the two main synthesis paths (state-machine primary vs rule primary)
and confirms the markdown renderer hits every required section.
"""
from __future__ import annotations

import json
from pathlib import Path

from conftest import FIXTURES
from slurm_doctor.collect import CollectedBundle, StdioFile
from slurm_doctor.diagnose import Diagnosis, ProposedFix, diagnose
from slurm_doctor.parse import load_rules
from slurm_doctor.report import render_json, render_markdown, write_report


def _missing_exec_bundle(tmp_path: Path) -> CollectedBundle:
    err = tmp_path / "stderr.txt"
    err.write_text((FIXTURES / "stderr_missing_exec.txt").read_text())
    submit = tmp_path / "submit_script.sh"
    submit.write_text("#!/bin/bash\necho hi\n./bogus_bin\n")  # no `set -e`
    b = CollectedBundle(
        jobid="2", cache_dir=str(tmp_path),
        state="FAILED", exit_code="127:0", derived_exit_code="0:0", reason="None",
        nodelist="c1", workdir="/data",
        req_mem="200M", elapsed="00:00:01", timelimit="00:01:00",
        submit_script_path=str(submit),
    )
    b.stderr = StdioFile(declared_path="/data/sd_demo_2.err",
                         cached_path=str(err), original_size=104, truncated=False)
    return b


def test_diagnose_missing_exec_rule_primary(tmp_path):
    bundle = _missing_exec_bundle(tmp_path)
    d = diagnose(bundle, load_rules())

    assert d.state_category == "failed_generic"
    assert "missing_executable" in {h["rule_id"] for h in d.rule_hits}
    assert "binary that doesn't exist" in d.root_cause or "doesn't exist" in d.root_cause
    assert d.confidence >= 0.85
    kinds = [f.fix_kind for f in d.proposed_fixes]
    assert "fix_path" in kinds
    # Script doesn't `set -e`, so add_set_eux is proposed as a follow-up.
    assert "add_set_eux" in kinds


def test_diagnose_timeout_state_primary():
    b = CollectedBundle(
        jobid="9", cache_dir="/tmp/9",
        state="TIMEOUT", exit_code="0:15", reason="TimeLimit",
        elapsed="00:01:30", timelimit="00:00:30",
        nodelist="c1", workdir="/data",
    )
    d = diagnose(b, load_rules())
    assert d.state_category == "timeout"
    assert "walltime" in d.tldr.lower()
    assert d.proposed_fixes
    assert d.proposed_fixes[0].fix_kind == "bump_time"


def test_diagnose_oom_state_primary():
    b = CollectedBundle(
        jobid="10", cache_dir="/tmp/10",
        state="OUT_OF_MEMORY", exit_code="0:9",
        req_mem="1G", max_rss="1100M",
        elapsed="00:00:10", nodelist="c1", workdir="/data",
    )
    d = diagnose(b, load_rules())
    assert d.state_category == "oom"
    assert d.proposed_fixes[0].fix_kind == "bump_memory"


def test_report_markdown_has_required_sections(tmp_path):
    bundle = _missing_exec_bundle(tmp_path)
    d = diagnose(bundle, load_rules())
    md = render_markdown(d, bundle)
    for header in ("## TL;DR", "## What ran", "## Why it failed",
                   "## Suggested fixes", "## Verification plan"):
        assert header in md, f"missing section: {header}"
    # Evidence must be quoted from stderr
    assert "No such file or directory" in md
    # Fix proposal shows up with confidence
    assert "fix_path" in md
    # Cache dir is referenced
    assert str(bundle.cache_dir) in md


def test_report_json_is_schema_versioned(tmp_path):
    bundle = _missing_exec_bundle(tmp_path)
    d = diagnose(bundle, load_rules())
    js = render_json(d, bundle)
    assert js["schema_version"] == 1
    assert js["jobid"] == "2"
    assert js["diagnosis"]["root_cause"]
    assert js["bundle"]["state"] == "FAILED"


def test_write_report_writes_both_files(tmp_path):
    bundle = _missing_exec_bundle(tmp_path)
    d = diagnose(bundle, load_rules())
    md, jsf = write_report(d, bundle, dest_root=tmp_path / "reports")
    assert md.exists() and jsf.exists()
    assert md.name == "report.md" and jsf.name == "report.json"
    # JSON round-trips
    parsed = json.loads(jsf.read_text())
    assert parsed["jobid"] == "2"
