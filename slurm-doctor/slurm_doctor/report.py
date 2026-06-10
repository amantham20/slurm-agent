"""Phase 3 — Report writers: report.md (human) + report.json (machine)."""

from __future__ import annotations

import dataclasses
import difflib
import json
import logging
from datetime import datetime
from pathlib import Path

from . import __version__
from .collect import Bundle
from .config import Config
from .diagnose import Diagnosis

log = logging.getLogger("slurm_doctor.report")

REPORT_SCHEMA_VERSION = 1


def fix_diff(original: str, patched: str, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=name,
            tofile=f"{name} (patched)",
        )
    )


_VERIFY_PLANS = {
    "bump_memory": [
        "Resubmit the patched script and watch `squeue -j <newid>`.",
        "After completion run `sacct -j <newid> --format=State,MaxRSS,ReqMem` - "
        "State should be COMPLETED and MaxRSS comfortably below the new request.",
    ],
    "bump_time": [
        "Resubmit and check `sacct -j <newid> --format=State,Elapsed,Timelimit`.",
        "If it times out again, the workload scales worse than x1.5 - profile it "
        "instead of bumping further.",
    ],
    "add_set_eux": [
        "Resubmit; the job should now fail FAST at the first real error with the "
        "offending line in stderr (bash -x style trace).",
    ],
    "prepend_modules": [
        "Resubmit and grep stderr for `module:` - the command-not-found error must be gone.",
        "Run `module avail <name>` on a compute node to confirm the module exists.",
    ],
    "fix_path": [
        "Resubmit and confirm stderr no longer shows `No such file or directory`.",
    ],
    "swap_mpi_launcher": [
        "Resubmit and check all ranks start: `sacct -j <newid> --format=JobID,State,NTasks`.",
        "Verify with a hostname rank-roll-call before the real workload.",
    ],
    "request_constraint": [
        "Resubmit and confirm placement: `sacct -j <newid> --format=NodeList` plus "
        "`scontrol show node <node> | grep AvailableFeatures`.",
    ],
    "pin_gpu_visible": [
        "Resubmit and check stderr for the CUDA init line; `nvidia-smi` inside the "
        "job should list the allocated GPU(s).",
    ],
    "add_requeue_guard": [
        "Resubmit; on the next transient failure check `sacct` for a REQUEUED entry "
        "instead of a hard failure.",
    ],
}


def render_markdown(diag: Diagnosis, bundle: Bundle, patches: dict[str, str]) -> str:
    """patches maps fix.kind -> patched script text (for the diff section)."""
    from .fix import _original_script_name

    p = bundle.parent
    script_name = _original_script_name(bundle)
    lines: list[str] = []
    add = lines.append

    add(f"# slurm-doctor report - job {diag.jobid} ({p.get('JobName', '?')})")
    add("")
    add(f"## TL;DR")
    add("")
    add(diag.tldr())
    add("")

    add("## What ran")
    add("")
    add("| field | value |")
    add("|---|---|")
    for k in ("JobName", "User", "Partition", "State", "ExitCode", "Reason",
              "Submit", "Start", "End", "Elapsed", "Timelimit", "ReqMem",
              "ReqCPUS", "NodeList", "WorkDir"):
        v = p.get(k, "")
        if v:
            add(f"| {k} | `{v}` |")
    steps = bundle.steps
    if steps:
        add("")
        add("Steps:")
        add("")
        add("| step | state | exit | MaxRSS | elapsed |")
        add("|---|---|---|---|---|")
        for s in steps:
            add(f"| {s.get('JobID')} | {s.get('State')} | {s.get('ExitCode')} "
                f"| {s.get('MaxRSS') or '-'} | {s.get('Elapsed')} |")
    if bundle.script:
        add("")
        add("Submit script:")
        add("")
        add("```bash")
        body = bundle.script.rstrip("\n")
        script_lines = body.splitlines()
        if len(script_lines) > 80:
            body = "\n".join(script_lines[:80]) + "\n# ... truncated ..."
        add(body)
        add("```")
    add("")

    add("## Why it failed")
    add("")
    add(f"**{diag.root_cause}**")
    add("")
    add(f"- category: `{diag.category}`  |  confidence: `{diag.confidence:.2f}`"
        + ("  |  diagnosed with LLM assist" if diag.used_llm else ""))
    if diag.rule_ids:
        add(f"- matched rules: {', '.join('`' + r + '`' for r in diag.rule_ids)}")
    add("")
    add("Evidence:")
    add("")
    for ev in diag.evidence[:12]:
        add(f"- {ev.cite()}" + (f" ({ev.match_count} matches)" if ev.match_count > 1 else ""))
    if diag.contributing:
        add("")
        add("Contributing factors:")
        add("")
        for c in diag.contributing:
            add(f"- {c}")
    add("")

    add("## Suggested fixes (ranked)")
    add("")
    if not diag.fixes:
        add("No automatic fix applies. See the hints above; this looks like "
            "something a human needs to decide.")
    for i, fix in enumerate(diag.fixes, 1):
        add(f"### {i}. {fix.title}")
        add("")
        add(f"`fix_kind={fix.kind}`  |  confidence `{fix.confidence:.2f}`"
            + ("  |  **requires --yes** (risky class)" if fix.requires_yes else ""))
        add("")
        add(fix.description)
        add("")
        if fix.kind in patches and bundle.script:
            diff = fix_diff(bundle.script, patches[fix.kind], script_name)
            if diff:
                add("```diff")
                add(diff.rstrip("\n"))
                add("```")
                add("")

    add("## Verification plan")
    add("")
    seen_steps: list[str] = []
    for fix in diag.fixes:
        for step in _VERIFY_PLANS.get(fix.kind, []):
            if step not in seen_steps:
                seen_steps.append(step)
    if not seen_steps:
        seen_steps = ["Re-run the job and compare `sacct` State/ExitCode against this report."]
    for i, step in enumerate(seen_steps, 1):
        add(f"{i}. {step}")
    add("")
    add(f"---")
    add(f"*generated by slurm-doctor {__version__} at "
        f"{datetime.now().isoformat(timespec='seconds')}; "
        f"raw artifacts: `{bundle.dir}`*")
    return "\n".join(lines) + "\n"


def write_reports(
    diag: Diagnosis, bundle: Bundle, cfg: Config, patches: dict[str, str] | None = None
) -> tuple[Path, Path]:
    patches = patches or {}
    outdir = cfg.job_report_dir(diag.jobid)
    outdir.mkdir(parents=True, exist_ok=True)

    md_path = outdir / "report.md"
    md_path.write_text(render_markdown(diag, bundle, patches))

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "slurm_doctor_version": __version__,
        "tldr": diag.tldr(),
        "diagnosis": diag.to_dict(),
        "job": bundle.manifest.get("parent_summary", {}),
        "artifacts_dir": str(bundle.dir),
    }
    json_path = outdir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s and %s", md_path, json_path)
    return md_path, json_path


def already_reported(jobid: str, cfg: Config) -> bool:
    """Used by sweep/hook to skip jobs that already have a current report."""
    f = cfg.job_report_dir(jobid) / "report.json"
    try:
        data = json.loads(f.read_text())
        return data.get("schema_version") == REPORT_SCHEMA_VERSION
    except (OSError, json.JSONDecodeError):
        return False
