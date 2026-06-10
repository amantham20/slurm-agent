"""Report writer tests: evidence citations, TL;DR, schema, idempotency."""

from __future__ import annotations

import json

from slurm_doctor.diagnose import diagnose
from slurm_doctor.fix import apply_fix
from slurm_doctor.report import already_reported, render_markdown, write_reports

from conftest import load_bundle


def test_report_md_structure(oom_bundle, cfg):
    diag = diagnose(oom_bundle, cfg)
    patches = {
        f.kind: apply_fix(f, oom_bundle.script, oom_bundle).patched
        for f in diag.fixes
        if apply_fix(f, oom_bundle.script, oom_bundle).changed
    }
    md = render_markdown(diag, oom_bundle, patches)
    for section in ("## TL;DR", "## What ran", "## Why it failed",
                    "## Suggested fixes (ranked)", "## Verification plan"):
        assert section in md
    # TL;DR is one line
    tldr = md.split("## TL;DR")[1].split("##")[0].strip()
    assert "\n" not in tldr
    # evidence must quote an actual log line with a location
    assert "`stderr:" in md and "exceeded memory limit" in md
    # diff against the original script, named after the script
    assert "```diff" in md and "--- oom.sh" in md
    assert "-#SBATCH --mem=50M" in md and "+#SBATCH --mem=" in md


def test_report_json_schema(oom_bundle, cfg):
    diag = diagnose(oom_bundle, cfg)
    md, js = write_reports(diag, oom_bundle, cfg)
    data = json.loads(js.read_text())
    assert data["schema_version"] == 1
    assert data["tldr"].startswith("Job 2:")
    d = data["diagnosis"]
    assert d["category"] == "memory"
    assert d["fixes"][0]["kind"] == "bump_memory"
    assert d["fixes"][0]["params"]["new_mem"] == "448M"
    assert d["evidence"] and all("line" in e for e in d["evidence"])
    assert data["job"]["JobName"] == "oom"


def test_write_reports_idempotent_and_tracked(oom_bundle, cfg):
    diag = diagnose(oom_bundle, cfg)
    assert not already_reported("2", cfg)
    write_reports(diag, oom_bundle, cfg)
    assert already_reported("2", cfg)
    # second run overwrites cleanly (no error, same path)
    md, _ = write_reports(diag, oom_bundle, cfg)
    assert md.exists()


def test_no_fix_report_still_has_evidence(cfg):
    bundle = load_bundle("6")  # python import error: hint, no scripted fix
    diag = diagnose(bundle, cfg)
    md = render_markdown(diag, bundle, {})
    assert "ModuleNotFoundError" in md
    assert "## Verification plan" in md


def test_diagnosis_confidences_and_categories(cfg):
    expected = {
        "2": ("memory", "bump_memory"),
        "3": ("time", "bump_time"),
        "8": ("environment", "prepend_modules"),
        "11": ("mpi", "swap_mpi_launcher"),
        "13": ("gpu", "request_constraint"),
    }
    for jobid, (cat, kind) in expected.items():
        diag = diagnose(load_bundle(jobid), cfg)
        assert diag.category == cat, f"job {jobid}: {diag.category} != {cat}"
        assert diag.fixes and diag.fixes[0].kind == kind
        assert 0 < diag.confidence <= 1
