"""slurm-doctor command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .config import Config, FAILURE_STATES
from .collect import Bundle, collect, wait_until_accounted
from .diagnose import Diagnosis, diagnose
from .fix import apply_fix, heal, write_patch
from .parse import RuleEngine
from .report import already_reported, write_reports
from .util import parse_since, run

log = logging.getLogger("slurm_doctor")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slurm-doctor",
        description="Autonomous SLURM job failure analyst: collect evidence, "
        "diagnose, report, patch and (carefully) resubmit.",
    )
    p.add_argument("--version", action="version", version=f"slurm-doctor {__version__}")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v for info, -vv for debug")
    p.add_argument("--cache-dir", help="raw artifact cache (default ~/.cache/slurm-doctor)")
    p.add_argument("--report-dir", help="report output dir (default /data/jobs/.slurm-doctor)")
    p.add_argument("--llm", action="store_true",
                   help="allow the Claude fallback when no rule matches "
                        "(also SLURM_DOCTOR_LLM=1; never on by default)")

    sub = p.add_subparsers(dest="command", required=True)

    def jobcmd(name: str, help_: str) -> argparse.ArgumentParser:
        c = sub.add_parser(name, help=help_)
        c.add_argument("jobid", help="SLURM job id")
        c.add_argument("--refresh", action="store_true",
                       help="re-collect even if cached artifacts exist")
        return c

    c = jobcmd("collect", "fetch and cache all evidence for a job (no analysis)")

    c = jobcmd("suggest", "diagnose and write report.md/report.json (no patches)")
    c.add_argument("--from-hook", action="store_true",
                   help="hook mode: wait for accounting to settle, skip if "
                        "already reported, print TL;DR for the slurmctld log")

    c = jobcmd("patch", "diagnose, write report AND patched script(s); no resubmit")

    c = jobcmd("heal", "diagnose, write report, patch the top fix and resubmit it")
    c.add_argument("--yes", action="store_true",
                   help="apply fixes below the confidence gate / outside the "
                        "allowlist / in the risky classes (MPI, GPU, constraints)")

    c = sub.add_parser("sweep", help="diagnose all recent failures")
    c.add_argument("--since", default="1 hour ago",
                   help="window start: '1 hour ago', '30 min ago', ISO time "
                        "(default: '1 hour ago')")
    c.add_argument("--refresh", action="store_true")
    c.add_argument("--states", default="FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL,BOOT_FAIL,DEADLINE",
                   help="comma list of sacct states to treat as failures")

    sub.add_parser("rules", help="list the loaded failure-signature rules")
    return p


def _config_from_args(args) -> Config:
    overrides = {}
    if args.cache_dir:
        overrides["cache_dir"] = Path(args.cache_dir)
    if args.report_dir:
        overrides["report_dir"] = Path(args.report_dir)
    if args.llm:
        overrides["llm_enabled"] = True
    return Config.from_env(**overrides)


def _diagnose_job(jobid: str, cfg: Config, refresh: bool) -> tuple[Bundle, Diagnosis]:
    bundle = collect(jobid, cfg, refresh=refresh)
    if not bundle.is_terminal:
        print(f"job {jobid} is {bundle.state or 'not in accounting yet'}; "
              "diagnosis needs a finished job", file=sys.stderr)
        raise SystemExit(3)
    return bundle, diagnose(bundle, cfg)


def _patches_for_report(diag: Diagnosis, bundle: Bundle) -> dict[str, str]:
    """Patched script text per fix kind, for the report's diff sections."""
    out: dict[str, str] = {}
    if not bundle.script:
        return out
    for fix in diag.fixes:
        res = apply_fix(fix, bundle.script, bundle)
        if res.changed:
            out[fix.kind] = res.patched
    return out


def cmd_collect(args, cfg: Config) -> int:
    bundle = collect(args.jobid, cfg, refresh=args.refresh)
    print(f"collected job {args.jobid} -> {bundle.dir}")
    for name, meta in sorted(bundle.manifest.get("artifacts", {}).items()):
        size = meta.get("bytes", 0)
        flags = " (missing)" if meta.get("missing") else ""
        flags += " (truncated)" if meta.get("truncated") else ""
        print(f"  {name:28s} {size:>10d} B{flags}")
    return 0


def cmd_suggest(args, cfg: Config) -> int:
    if getattr(args, "from_hook", False):
        if already_reported(args.jobid, cfg) and not args.refresh:
            log.info("job %s already has a report; skipping", args.jobid)
            return 0
        wait_until_accounted(args.jobid, cfg)
    bundle, diag = _diagnose_job(args.jobid, cfg, args.refresh)
    md, js = write_reports(diag, bundle, cfg, _patches_for_report(diag, bundle))
    print(diag.tldr())
    print(f"report: {md}")
    return 0


def cmd_patch(args, cfg: Config) -> int:
    bundle, diag = _diagnose_job(args.jobid, cfg, args.refresh)
    patches = _patches_for_report(diag, bundle)
    md, _ = write_reports(diag, bundle, cfg, patches)
    print(diag.tldr())
    print(f"report: {md}")
    if not bundle.script:
        print("submit script unavailable; no patch written", file=sys.stderr)
        return 2
    wrote_any = False
    for fix in diag.fixes:
        res = apply_fix(fix, bundle.script, bundle)
        if res.downgraded:
            print(f"  [{fix.kind}] not patched: {'; '.join(res.notes)}")
            continue
        if not res.changed:
            print(f"  [{fix.kind}] not applicable: {'; '.join(res.notes)}")
            continue
        path = write_patch(res, bundle, cfg)
        wrote_any = True
        print(f"  [{fix.kind}] patched script: {path}")
    if not wrote_any:
        print("no patch files written")
    return 0


def cmd_heal(args, cfg: Config) -> int:
    bundle, diag = _diagnose_job(args.jobid, cfg, args.refresh)
    md, _ = write_reports(diag, bundle, cfg, _patches_for_report(diag, bundle))
    print(diag.tldr())
    print(f"report: {md}")
    outcome = heal(diag, bundle, cfg, yes=args.yes)
    if outcome.healed:
        print(f"healed: {outcome.reason}")
        print(f"patched script: {outcome.patch_file}")
        return 0
    print(f"not healed: {outcome.reason}", file=sys.stderr)
    return 4


def _sweep_jobids(cfg: Config, since_epoch: int, states: set[str]) -> list[str]:
    # Prefer slurmrestd when reachable; sacct otherwise.
    from .restclient import RestClient, rest_failed_jobids

    client = RestClient.from_env(cfg.restd_url)
    if client:
        ids = rest_failed_jobids(client, since_epoch, states)
        if ids is not None:
            log.info("sweep: slurmrestd returned %d failed job(s)", len(ids))
            return ids
        log.info("sweep: slurmrestd unusable, falling back to sacct")
    state_flags = ",".join(sorted({_SACCT_STATE_CODES.get(s, s) for s in states}))
    start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(since_epoch))
    r = run(["sacct", "-X", "--noheader", "--parsable2", "--format=JobIDRaw,State",
             f"--state={state_flags}", f"--starttime={start}", "--endtime=now"],
            timeout=cfg.cmd_timeout)
    ids = []
    for ln in r.stdout.splitlines():
        cols = ln.split("|")
        if cols and cols[0].isdigit():
            ids.append(cols[0])
    return ids


_SACCT_STATE_CODES = {
    "FAILED": "F", "TIMEOUT": "TO", "OUT_OF_MEMORY": "OOM",
    "NODE_FAIL": "NF", "BOOT_FAIL": "BF", "DEADLINE": "DL",
    "CANCELLED": "CA", "PREEMPTED": "PR",
}


def cmd_sweep(args, cfg: Config) -> int:
    since = parse_since(args.since)
    states = {s.strip().upper() for s in args.states.split(",") if s.strip()}
    bad = states - set(FAILURE_STATES)
    if bad:
        print(f"unknown failure states: {', '.join(sorted(bad))}", file=sys.stderr)
        return 2
    ids = _sweep_jobids(cfg, int(since.timestamp()), states)
    if not ids:
        print(f"no failed jobs since {since.isoformat(timespec='seconds')}")
        return 0
    done = skipped = failed = 0
    for jobid in ids:
        if already_reported(jobid, cfg) and not args.refresh:
            skipped += 1
            continue
        try:
            bundle = collect(jobid, cfg, refresh=args.refresh)
            if not bundle.is_terminal:
                skipped += 1
                continue
            diag = diagnose(bundle, cfg)
            md, _ = write_reports(diag, bundle, cfg, _patches_for_report(diag, bundle))
            print(f"{diag.tldr()}\n  -> {md}")
            done += 1
        except Exception as exc:  # one broken job must not kill the sweep
            log.error("sweep: job %s failed: %s", jobid, exc)
            failed += 1
    print(f"sweep: {done} reported, {skipped} skipped (already reported/not final), "
          f"{failed} errored")
    return 0 if failed == 0 else 5


def cmd_rules(args, cfg: Config) -> int:
    engine = RuleEngine.load()
    print(f"{len(engine.rules)} rules loaded\n")
    print(f"{'id':<26} {'category':<15} {'fix_kind':<20} conf  source")
    for r in engine.rules:
        print(f"{r.id:<26} {r.category:<15} {str(r.fix_kind or '-'):<20} "
              f"{r.confidence:<5.2f} {r.source_file}")
    return 0


_COMMANDS = {
    "collect": cmd_collect,
    "suggest": cmd_suggest,
    "patch": cmd_patch,
    "heal": cmd_heal,
    "sweep": cmd_sweep,
    "rules": cmd_rules,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    cfg = _config_from_args(args)
    try:
        return _COMMANDS[args.command](args, cfg)
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError) as exc:
        print(f"slurm-doctor: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
