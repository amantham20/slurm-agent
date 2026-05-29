"""`slurm-doctor` CLI: suggest, patch, heal.

- ``suggest <jobid>``  collect + diagnose + report; never modifies anything else.
- ``patch <jobid>``    suggest + write a patched script for every safe fix kind.
- ``heal <jobid>``     patch + sbatch the top-ranked auto-applicable fix.

The heal gate matches the spec: confidence >= 0.85 AND fix_kind in the safe
allowlist (default ``bump_memory``, ``bump_time``, ``add_set_eux``) unless
``--yes`` is passed. Every resubmission carries
``--comment=slurm-doctor:fix=<kind>:parent=<jobid>`` so chains are traceable
and a future fix-loop guard can refuse to heal a job already healed twice.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from . import __version__
from ._shell import default_runner
from .collect import collect
from .diagnose import SAFE_FIX_KINDS, Diagnosis, ProposedFix, diagnose
from .fix import FixOutcome, apply_fix, write_patched_script
from .parse import load_rules
from .report import write_report

log = logging.getLogger("slurm_doctor.cli")

DEFAULT_REPORTS_DIR = Path("/data/jobs/.slurm-doctor")
COMMENT_PREFIX = "slurm-doctor:fix="
MAX_HEALS_PER_CHAIN = 2


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_suggest(args) -> int:
    bundle, diag, paths = _collect_diagnose_report(args)
    md, _js = paths
    if getattr(args, "hook", False):
        # Hook-triggered: emit a single compact line meant to surface in logs.
        print(f"slurm-doctor[hook]: {diag.tldr} -> {md}")
        return 0
    print(diag.tldr)
    print(f"report: {md}")
    return 0


def cmd_patch(args) -> int:
    bundle, diag, (md, _js) = _collect_diagnose_report(args)
    allow = _allow_set(args)
    if bundle.submit_script_path is None:
        print(f"slurm-doctor: no submit script in bundle for job {bundle.jobid}; "
              "cannot patch", file=sys.stderr)
        return 2
    original = Path(bundle.submit_script_path).read_text()

    written: list[tuple[ProposedFix, FixOutcome, Path]] = []
    for fix in diag.proposed_fixes:
        if not _gate_allows(fix, allow, yes=args.yes):
            log.info("skipping fix %s: gated (conf %.2f, yes=%s)",
                     fix.fix_kind, fix.confidence, args.yes)
            continue
        out = apply_fix(fix.fix_kind, original, bundle)
        if out.patched_text is None:
            log.info("fix %s skipped: %s", fix.fix_kind, out.skipped_reason)
            continue
        dst = write_patched_script(bundle, out)
        fix.patched_script_path = str(dst)
        fix.diff = out.diff
        written.append((fix, out, dst))

    # Re-render the report so it includes real diffs / patched paths.
    md, _js = write_report(diag, bundle, dest_root=args.reports_dir)

    print(diag.tldr)
    print(f"report: {md}")
    if not written:
        print("no patches written (all fixes gated or unable to apply).")
        return 1
    for fix, _out, dst in written:
        print(f"patched ({fix.fix_kind}, conf {fix.confidence:.2f}): {dst}")
    return 0


# States sacct should treat as failures worth sweeping (abbreviations).
SWEEP_STATES = "F,TO,OOM,NF,BF,DL"


def _normalise_since(since: str) -> str:
    """Translate friendly '<N> hour(s) ago' / '<N> day(s) ago' into sacct's
    now-<N><unit> syntax; otherwise pass the value through unchanged so callers
    can use any native sacct time (e.g. 2026-05-29T10:00:00, now-90minutes)."""
    m = re.match(r"^\s*(\d+)\s+(second|minute|hour|day|week)s?\s+ago\s*$", since, re.I)
    if not m:
        return since
    n, unit = m.group(1), m.group(2).lower()
    return f"now-{n}{unit}s"


def cmd_sweep(args) -> int:
    since = _normalise_since(args.since)
    # sacct's --state filter is a no-op unless an explicit --endtime is given
    # alongside --starttime; without it the time window collapses and nothing
    # matches. Pin endtime to the configurable --until (default now).
    r = default_runner(
        ["sacct", "-X", "-n", "--parsable2", f"--starttime={since}",
         f"--endtime={args.until}", f"--state={SWEEP_STATES}",
         "--format=JobID,State"],
        timeout=30,
    )
    if r.missing or r.returncode != 0:
        print(f"slurm-doctor: sacct failed (rc={r.returncode}): {r.stderr[:200]}",
              file=sys.stderr)
        return 2

    jobids: list[str] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        jid = line.split("|")[0]
        # Skip array/het sub-components that sacct may emit (e.g. 12_3, 12+0);
        # keep the plain or array-task ids the collector accepts.
        if re.match(r"^[0-9]+(_[0-9]+)?$", jid):
            jobids.append(jid)

    reports_root = Path(args.reports_dir)
    rules = load_rules(args.rules_dir)
    n_new = n_skip = n_err = 0
    for jid in jobids:
        if (reports_root / jid / "report.md").exists() and not args.force:
            n_skip += 1
            continue
        try:
            bundle = collect(jid, cache_root=args.cache_dir)
            diag = diagnose(bundle, rules, use_llm=_llm_enabled(args))
            md, _ = write_report(diag, bundle, dest_root=args.reports_dir)
            n_new += 1
            print(f"[{jid}] {diag.tldr}")
        except Exception as e:  # noqa: BLE001 - keep sweeping past one bad job
            n_err += 1
            log.warning("sweep: job %s failed: %r", jid, e)
    print(f"sweep since '{since}': {len(jobids)} failed job(s); "
          f"{n_new} reported, {n_skip} already done, {n_err} errored.")
    return 0


def cmd_heal(args) -> int:
    bundle, diag, (md, _js) = _collect_diagnose_report(args)
    allow = _allow_set(args)

    # Fix-loop guard: refuse if this job's ancestry already used N healings.
    chain_count = _count_chain_healings(bundle, runner=default_runner)
    if chain_count >= MAX_HEALS_PER_CHAIN:
        print(
            f"slurm-doctor: job {bundle.jobid} ancestry already healed "
            f"{chain_count} times (>= MAX={MAX_HEALS_PER_CHAIN}); refusing to "
            "heal again to avoid fix loops. Use --yes-loop to override.",
            file=sys.stderr,
        )
        return 3

    if bundle.submit_script_path is None:
        print(f"slurm-doctor: no submit script for job {bundle.jobid}; "
              "cannot heal", file=sys.stderr)
        return 2
    original = Path(bundle.submit_script_path).read_text()

    # Apply every safe fix that passes the gate to populate diffs in the
    # report, but heal only the highest-confidence one.
    ranked = sorted(diag.proposed_fixes, key=lambda f: -f.confidence)
    chosen: tuple[ProposedFix, FixOutcome, Path] | None = None
    for fix in ranked:
        if not _gate_allows(fix, allow, yes=args.yes):
            continue
        out = apply_fix(fix.fix_kind, original, bundle)
        if out.patched_text is None:
            log.info("heal: fix %s skipped (%s)", fix.fix_kind, out.skipped_reason)
            continue
        dst = write_patched_script(bundle, out)
        fix.patched_script_path = str(dst)
        fix.diff = out.diff
        if chosen is None:
            chosen = (fix, out, dst)

    md, _js = write_report(diag, bundle, dest_root=args.reports_dir)

    print(diag.tldr)
    print(f"report: {md}")
    if chosen is None:
        print("nothing safe to heal (every fix was gated or unable to apply).")
        return 1

    fix, _out, dst = chosen
    new_jobid = _sbatch(
        dst,
        comment=f"{COMMENT_PREFIX}{fix.fix_kind}:parent={bundle.jobid}",
        chdir=bundle.workdir,
        dry_run=args.dry_run,
    )
    if new_jobid is None:
        print(f"slurm-doctor: sbatch failed for {dst}", file=sys.stderr)
        return 4
    print(f"healed: applied {fix.fix_kind} (conf {fix.confidence:.2f}) -> {dst}")
    print(f"sbatch: new jobid={new_jobid}")
    return 0


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------

def _llm_enabled(args) -> bool:
    return bool(getattr(args, "llm", False)) or os.environ.get("SLURM_DOCTOR_LLM") == "1"


def _collect_diagnose_report(args):
    bundle = collect(args.jobid, cache_root=args.cache_dir)
    rules = load_rules(args.rules_dir)
    diag = diagnose(bundle, rules, use_llm=_llm_enabled(args))
    md, js = write_report(diag, bundle, dest_root=args.reports_dir)
    return bundle, diag, (md, js)


def _allow_set(args) -> set[str]:
    """Allowlist of fix_kinds heal/patch may auto-apply without --yes."""
    if args.allow:
        return set(_split_csv(args.allow))
    return set(SAFE_FIX_KINDS)


def _gate_allows(fix: ProposedFix, allow: set[str], *, yes: bool) -> bool:
    if yes:
        return True
    return fix.fix_kind in allow and fix.confidence >= 0.85


def _split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _sbatch(script: Path, *, comment: str, chdir: str | None = None,
            dry_run: bool = False) -> str | None:
    argv = ["sbatch", "--parsable", "--comment", comment]
    if chdir:
        argv += ["--chdir", chdir]
    argv.append(str(script))
    if dry_run:
        print(f"DRY RUN: {' '.join(argv)}")
        return "DRYRUN"
    r = default_runner(argv, timeout=30)
    if r.missing or r.returncode != 0:
        log.error("sbatch failed rc=%s stderr=%s", r.returncode, r.stderr[:200])
        return None
    out = (r.stdout or "").strip()
    # `--parsable` returns: "<jobid>;<cluster>" or just "<jobid>".
    return out.split(";")[0].split()[0] if out else None


_COMMENT_RX = re.compile(r"slurm-doctor:fix=([\w_]+):parent=(\d+)")


def _count_chain_healings(bundle, *, runner) -> int:
    """Count how many ancestor jobs in this chain were already healed.

    Walks comment fields backwards via sacct until we run out of parents.
    """
    seen: set[str] = set()
    cur = bundle.jobid
    count = 0
    for _ in range(MAX_HEALS_PER_CHAIN + 2):  # bounded walk
        if cur in seen:
            break
        seen.add(cur)
        r = runner(["sacct", "-j", cur, "-X", "--parsable2",
                    "--format=Comment"], timeout=15)
        if r.missing or r.returncode != 0 or not r.stdout.strip():
            break
        # First non-header line.
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        if len(lines) < 2:
            break
        comment = lines[1].strip()
        m = _COMMENT_RX.search(comment)
        if not m:
            break
        count += 1
        cur = m.group(2)  # parent jobid; walk further back
    return count


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="slurm-doctor",
                                 description="Diagnose, report on, and heal failed SLURM jobs.")
    ap.add_argument("--version", action="version", version=f"slurm-doctor {__version__}")
    ap.add_argument("--log-level", default=os.environ.get("SLURM_DOCTOR_LOG", "WARNING"))
    ap.add_argument("--cache-dir", default=None,
                    help="Override the artifact cache root (default ~/.cache/slurm-doctor)")
    ap.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR),
                    help="Where to write report.md / report.json")
    ap.add_argument("--rules-dir", default=None,
                    help="Override the YAML rules dir (default <pkg>/rules)")
    ap.add_argument("--allow", default=None,
                    help="Comma-separated allowlist of fix_kinds heal/patch may "
                         "auto-apply (default: bump_memory,bump_time,add_set_eux)")
    ap.add_argument("--yes", action="store_true",
                    help="Override the gate; apply even non-safe fix kinds.")
    ap.add_argument("--dry-run", action="store_true",
                    help="For heal: don't actually sbatch the patched script.")
    ap.add_argument("--hook", action="store_true",
                    help="Mark this as a hook-triggered run (compact stdout line).")
    ap.add_argument("--llm", action="store_true",
                    help="Enable the LLM fallback (Layer 3) when no rule fires. "
                         "Off by default; also enabled by SLURM_DOCTOR_LLM=1.")

    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("suggest", "patch", "heal"):
        sp = sub.add_parser(name, help=f"{name} action")
        sp.add_argument("jobid", help="The SLURM job id to diagnose")
    sweep = sub.add_parser("sweep", help="diagnose every recent failed job")
    sweep.add_argument("--since", default="1 hour ago",
                       help="sacct start time: '1 hour ago', 'now-90minutes', "
                            "or an absolute 2026-05-29T10:00:00")
    sweep.add_argument("--until", default="now",
                       help="sacct end time (default now); required by sacct's "
                            "--state filter")
    sweep.add_argument("--force", action="store_true",
                       help="re-report jobs that already have a report")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    fn = {"suggest": cmd_suggest, "patch": cmd_patch, "heal": cmd_heal,
          "sweep": cmd_sweep}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
