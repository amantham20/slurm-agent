"""Phase 4 — Fix & resubmit.

Takes the original submit script plus a ProposedFix and produces a surgically
patched copy. Patches are idempotent (marker comments guard inserted blocks)
and minimal (a >20% line-change rate downgrades the fix to suggest-only).

The patched script is written next to the original as
``<original>.fix<N>.sh`` — the original is never modified.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import logging
import os
import re
import time
from pathlib import Path

from .collect import Bundle
from .config import Config
from .diagnose import Diagnosis, ProposedFix
from .util import run

log = logging.getLogger("slurm_doctor.fix")

MARKER = "# [slurm-doctor]"
MAX_CHANGE_RATIO = 0.20

LMOD_INIT_BLOCK = f"""{MARKER} prepend_modules: initialise the module system in batch shells
if ! command -v module >/dev/null 2>&1; then
    for _lmod_init in /etc/profile.d/z00_lmod.sh /etc/profile.d/lmod.sh \\
                      /usr/share/lmod/lmod/init/bash /usr/local/lmod/lmod/init/bash; do
        [ -r "$_lmod_init" ] && . "$_lmod_init" && break
    done
fi"""


@dataclasses.dataclass
class PatchResult:
    fix: ProposedFix
    original: str
    patched: str
    notes: list[str] = dataclasses.field(default_factory=list)
    applicable: bool = True
    downgraded: bool = False  # True when the diff would exceed MAX_CHANGE_RATIO

    @property
    def changed(self) -> bool:
        return self.applicable and self.patched != self.original


# --------------------------------------------------------------------------
# Script surgery helpers
# --------------------------------------------------------------------------


def _directive_span(lines: list[str]) -> tuple[int, int]:
    """Return (first, last+1) indices of the leading #SBATCH block."""
    start = 1 if lines and lines[0].startswith("#!") else 0
    i = start
    last = start
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("#SBATCH"):
            last = i + 1
            i += 1
        elif s == "" or s.startswith("#"):
            i += 1  # comments/blank lines inside the header don't end it
        else:
            break
    return start, max(last, start)


def set_sbatch_option(lines: list[str], opts: list[str], value: str, notes: list[str]) -> list[str]:
    """Replace the value of the first matching sbatch option, or insert a new
    directive. *opts* lists synonyms, longest first (e.g. ['--mem'])."""
    pat = re.compile(
        r"^(?P<head>\s*#SBATCH\s+(?:.*\s)??(?P<opt>" + "|".join(re.escape(o) for o in opts) + r"))(?P<sep>[= ]\s*)(?P<val>\S+)"
    )
    out = list(lines)
    for i, line in enumerate(out):
        m = pat.match(line)
        if m:
            if m.group("val") == value:
                notes.append(f"{m.group('opt')} already set to {value}; nothing to do")
                return out
            out[i] = pat.sub(lambda mm: f"{mm.group('head')}{mm.group('sep')}{value}", line, count=1)
            notes.append(f"replaced {m.group('opt')} {m.group('val')} -> {value}")
            return out
    _, end = _directive_span(out)
    out.insert(end, f"#SBATCH {opts[0]}={value}")
    notes.append(f"inserted #SBATCH {opts[0]}={value}")
    return out


def insert_after_directives(lines: list[str], block: str, guard: str, notes: list[str]) -> list[str]:
    """Insert *block* right after the #SBATCH header unless *guard* already
    appears in the script (idempotency)."""
    joined = "\n".join(lines)
    if guard in joined:
        notes.append("already applied; nothing to do")
        return list(lines)
    out = list(lines)
    _, end = _directive_span(out)
    out[end:end] = block.splitlines() + [""]
    notes.append(f"inserted block after the #SBATCH header (line {end + 1})")
    return out


# --------------------------------------------------------------------------
# Fix kind implementations
# --------------------------------------------------------------------------


def _apply_bump_memory(lines, fix, bundle, notes):
    return set_sbatch_option(lines, ["--mem"], fix.params["new_mem"], notes)


def _apply_bump_time(lines, fix, bundle, notes):
    return set_sbatch_option(lines, ["--time", "-t"], fix.params["new_time"], notes)


def _apply_add_set_eux(lines, fix, bundle, notes):
    if any(re.match(r"^\s*set\s+-[a-zA-Z]*e", ln) for ln in lines):
        notes.append("script already sets -e; nothing to do")
        return list(lines)
    block = f"{MARKER} add_set_eux: stop at the first failing command\nset -euo pipefail"
    return insert_after_directives(lines, block, "set -euo pipefail", notes)


def _apply_prepend_modules(lines, fix, bundle, notes):
    out = insert_after_directives(lines, LMOD_INIT_BLOCK, "prepend_modules:", notes)
    mods = fix.params.get("modules") or ""
    if isinstance(mods, str):
        mods = [m for m in re.split(r"[,\s]+", mods) if m]
    inserted_at = None
    for i, ln in enumerate(out):
        if "prepend_modules:" in ln:
            inserted_at = i
            break
    if mods and inserted_at is not None:
        load_lines = [f"module load {m}" for m in mods if f"module load {m}" not in out]
        end = inserted_at
        while end < len(out) and out[end].strip() != "":
            end += 1
        out[end:end] = load_lines
        if load_lines:
            notes.append(f"added module load for: {', '.join(mods)}")
    return out


def _apply_fix_path(lines, fix, bundle, notes):
    missing = fix.params.get("missing_path", "")
    if not missing:
        notes.append("no missing path captured from the logs")
        return None
    workdir = bundle.workdir or "."
    base = os.path.basename(missing)
    candidates: list[str] = []
    try:
        seen = 0
        for root, dirs, files in os.walk(workdir):
            if root.count(os.sep) - workdir.count(os.sep) > 3:
                dirs[:] = []
                continue
            for f in files:
                seen += 1
                if seen > 5000:
                    raise StopIteration
                if f == base:
                    candidates.append(os.path.join(root, f))
    except StopIteration:
        pass
    except OSError as exc:
        notes.append(f"cannot search WorkDir: {exc}")
        return None
    if len(candidates) != 1:
        notes.append(
            f"found {len(candidates)} files named {base!r} under {workdir}; "
            "cannot rewrite the path unambiguously"
        )
        return None
    replacement = candidates[0]
    out = []
    hits = 0
    for ln in lines:
        if missing in ln and not ln.strip().startswith("#"):
            out.append(ln.replace(missing, replacement))
            hits += 1
        else:
            out.append(ln)
    if not hits:
        notes.append(f"{missing!r} not found in the script body")
        return None
    notes.append(f"rewrote {missing} -> {replacement} ({hits} lines)")
    return out


def _apply_swap_mpi_launcher(lines, fix, bundle, notes):
    pat = re.compile(r"^(?P<indent>\s*)(?P<launcher>mpirun|mpiexec|orterun|prterun)\b(?P<rest>.*)$")
    np_pat = re.compile(r"\s+(?:-n|-np|--np)\s+\d+")
    out = []
    hits = 0
    for ln in lines:
        m = pat.match(ln)
        if m and not ln.strip().startswith("#"):
            rest = np_pat.sub("", m.group("rest"))
            out.append(f"{m.group('indent')}srun --mpi=pmix{rest}")
            hits += 1
        else:
            out.append(ln)
    if not hits:
        notes.append("no mpirun/mpiexec invocation found")
        return None
    notes.append(
        f"replaced {hits} mpirun/mpiexec call(s) with `srun --mpi=pmix`; explicit "
        "-n/-np dropped so the rank count follows --ntasks"
    )
    return out


def _apply_request_constraint(lines, fix, bundle, notes):
    out = list(lines)
    feat = fix.params.get("constraint")
    if feat:
        out = set_sbatch_option(out, ["--constraint", "-C"], feat, notes)
    if fix.params.get("partition"):
        out = set_sbatch_option(out, ["--partition", "-p"], fix.params["partition"], notes)
    if fix.params.get("gres"):
        out = set_sbatch_option(out, ["--gres"], fix.params["gres"], notes)
    if out == list(lines):
        notes.append("no constraint/partition/gres parameters resolved")
        return None
    return out


def _apply_add_requeue_guard(lines, fix, bundle, notes):
    out = list(lines)
    if not any(re.search(r"#SBATCH\s+--requeue\b", ln) for ln in out):
        _, end = _directive_span(out)
        out[end:end] = [f"{MARKER} add_requeue_guard: survive transient infra failures",
                        "#SBATCH --requeue",
                        "#SBATCH --open-mode=append"]
        notes.append("added --requeue and --open-mode=append directives")
    else:
        notes.append("--requeue already present; nothing to do")
    return out


def _apply_pin_gpu_visible(lines, fix, bundle, notes):
    out = []
    commented = 0
    for ln in lines:
        if re.match(r"^\s*(export\s+)?CUDA_VISIBLE_DEVICES=", ln):
            out.append(f"{MARKER} pin_gpu_visible: let SLURM manage GPU visibility\n# {ln}")
            commented += 1
        else:
            out.append(ln)
    out2 = "\n".join(out).splitlines()  # flatten the two-line replacement
    if not any(re.search(r"#SBATCH\s+(--gpus-per-task|--gres=gpu)", ln) for ln in out2):
        out2 = set_sbatch_option(out2, ["--gpus-per-task"], fix.params.get("gpus", "1"), notes)
    if commented:
        notes.append(f"commented out {commented} manual CUDA_VISIBLE_DEVICES assignment(s)")
    return out2


_APPLIERS = {
    "bump_memory": _apply_bump_memory,
    "bump_time": _apply_bump_time,
    "add_set_eux": _apply_add_set_eux,
    "prepend_modules": _apply_prepend_modules,
    "fix_path": _apply_fix_path,
    "swap_mpi_launcher": _apply_swap_mpi_launcher,
    "request_constraint": _apply_request_constraint,
    "add_requeue_guard": _apply_add_requeue_guard,
    "pin_gpu_visible": _apply_pin_gpu_visible,
}


def apply_fix(fix: ProposedFix, script_text: str, bundle: Bundle) -> PatchResult:
    """Produce the patched script text for one fix (in memory)."""
    applier = _APPLIERS.get(fix.kind)
    notes: list[str] = []
    if applier is None:
        return PatchResult(fix, script_text, script_text,
                           [f"no applier for fix kind {fix.kind}"], applicable=False)
    lines = script_text.splitlines()
    new_lines = applier(lines, fix, bundle, notes)
    if new_lines is None:
        return PatchResult(fix, script_text, script_text, notes, applicable=False)
    patched = "\n".join(new_lines) + ("\n" if script_text.endswith("\n") else "")

    # Minimal-diff guard: a "fix" that REWRITES the script is not a fix.
    # Rewritten = existing lines replaced or deleted; pure insertions of
    # marker-guarded blocks don't touch existing lines and get a looser cap.
    sm = difflib.SequenceMatcher(a=lines, b=new_lines, autojunk=False)
    rewritten = inserted = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("replace", "delete"):
            rewritten += i2 - i1
        if op in ("replace", "insert"):
            inserted += j2 - j1
    total = max(len(lines), 1)
    # An absolute floor of 2 lines keeps surgical one-line fixes legal even in
    # very short scripts, where any change exceeds 20%.
    rewrite_cap = max(2, MAX_CHANGE_RATIO * total)
    insert_cap = max(8, 0.5 * total)
    if rewritten > rewrite_cap or inserted > insert_cap:
        notes.append(
            f"patch would rewrite {rewritten}/{total} lines and insert {inserted} "
            f"(caps: {rewrite_cap:.0f} rewritten, {insert_cap:.0f} inserted); "
            "downgrading to suggest-only"
        )
        return PatchResult(fix, script_text, patched, notes, applicable=False, downgraded=True)
    return PatchResult(fix, script_text, patched, notes)


# --------------------------------------------------------------------------
# Writing patches & resubmitting
# --------------------------------------------------------------------------


def _original_script_name(bundle: Bundle) -> str:
    cmd = bundle.scontrol_field("Command")
    if cmd and cmd not in ("(null)", "None"):
        return Path(cmd.split()[0]).name
    name = bundle.parent.get("JobName") or f"job{bundle.jobid}"
    return name if name.endswith(".sh") else f"{name}.sh"


def patch_path(bundle: Bundle, cfg: Config) -> Path:
    """Next free <workdir>/<original>.fix<N>.sh (never overwrites)."""
    workdir = Path(bundle.workdir or ".")
    base = _original_script_name(bundle)
    stem = base[:-3] if base.endswith(".sh") else base
    n = 1
    while (workdir / f"{stem}.fix{n}.sh").exists():
        n += 1
    return workdir / f"{stem}.fix{n}.sh"


def write_patch(result: PatchResult, bundle: Bundle, cfg: Config) -> Path | None:
    if not result.changed:
        return None
    path = patch_path(bundle, cfg)
    path.write_text(result.patched)
    path.chmod(0o755)
    log.info("wrote patched script %s (%s)", path, result.fix.kind)
    return path


# --------------------------------------------------------------------------
# Heal: gates, loop protection, resubmission
# --------------------------------------------------------------------------


@dataclasses.dataclass
class HealOutcome:
    healed: bool
    reason: str
    new_jobid: str | None = None
    patch_file: str | None = None
    fix_kind: str | None = None


def _chain_file(cfg: Config) -> Path:
    return cfg.report_dir / "chain.json"


def _load_chain(cfg: Config) -> dict:
    try:
        return json.loads(_chain_file(cfg).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _record_chain(cfg: Config, child: str, parent: str, kind: str, script: str) -> None:
    chain = _load_chain(cfg)
    chain[str(child)] = {
        "parent": str(parent),
        "fix": kind,
        "script": script,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _chain_file(cfg).parent.mkdir(parents=True, exist_ok=True)
    _chain_file(cfg).write_text(json.dumps(chain, indent=2))


_COMMENT_RE = re.compile(r"slurm-doctor:fix=(?P<kind>[\w-]+):parent=(?P<parent>\d+)")


def heal_chain_depth(bundle: Bundle, cfg: Config) -> int:
    """How many slurm-doctor heals are already in this job's ancestry.

    Uses the durable --comment trail (AccountingStoreFlags=job_comment) and
    the local chain ledger, whichever goes deeper.
    """
    depth = 0
    seen: set[str] = set()
    jobid = str(bundle.jobid)
    comment = bundle.parent.get("Comment") or ""
    chain = _load_chain(cfg)
    while jobid not in seen:
        seen.add(jobid)
        m = _COMMENT_RE.search(comment)
        entry = chain.get(jobid)
        parent = None
        if m:
            parent = m.group("parent")
        elif entry:
            parent = entry.get("parent")
        if not parent or str(parent) == jobid:
            break
        depth += 1
        jobid = str(parent)
        r = run(["sacct", "-j", jobid, "-X", "--noheader", "--parsable2",
                 "--format=Comment"], timeout=cfg.cmd_timeout)
        comment = r.stdout.strip().splitlines()[0] if r.ok and r.stdout.strip() else ""
    return depth


def heal(
    diag: Diagnosis,
    bundle: Bundle,
    cfg: Config,
    yes: bool = False,
) -> HealOutcome:
    """Patch the top fix and resubmit it, subject to the safety gates."""
    if not diag.fixes:
        return HealOutcome(False, "no fixes proposed; nothing to heal")
    if not bundle.script:
        return HealOutcome(False, "submit script could not be recovered; cannot patch")

    depth = heal_chain_depth(bundle, cfg)
    if depth >= cfg.max_heal_chain:
        return HealOutcome(
            False,
            f"refusing to heal: this job is already {depth} heal(s) deep "
            f"(max {cfg.max_heal_chain}). A fix loop means the diagnosis is "
            "wrong - a human should look at it.",
        )

    fix = diag.fixes[0]
    if not yes:
        if fix.requires_yes:
            return HealOutcome(
                False,
                f"fix `{fix.kind}` touches MPI/GPU/constraints and always needs "
                "--yes; rerun with --yes to apply it.",
                fix_kind=fix.kind,
            )
        if fix.confidence < cfg.heal_confidence:
            return HealOutcome(
                False,
                f"confidence {fix.confidence:.2f} is below the auto-heal gate "
                f"({cfg.heal_confidence:.2f}); rerun with --yes to apply anyway.",
                fix_kind=fix.kind,
            )
        if fix.kind not in cfg.autofix_kinds:
            return HealOutcome(
                False,
                f"fix `{fix.kind}` is not in the auto-heal allowlist "
                f"({', '.join(cfg.autofix_kinds)}); rerun with --yes.",
                fix_kind=fix.kind,
            )

    result = apply_fix(fix, bundle.script, bundle)
    if result.downgraded:
        return HealOutcome(False, "patch too large, downgraded to suggest: "
                           + "; ".join(result.notes), fix_kind=fix.kind)
    if not result.changed:
        return HealOutcome(False, "fix not applicable: " + "; ".join(result.notes),
                           fix_kind=fix.kind)

    path = write_patch(result, bundle, cfg)
    workdir = bundle.workdir or "."

    argv = ["sbatch", f"--comment=slurm-doctor:fix={fix.kind}:parent={bundle.jobid}"]
    r = run(["squeue", "-j", str(bundle.jobid), "-h", "-o", "%T"], timeout=cfg.cmd_timeout)
    if r.ok and "PENDING" in r.stdout:
        argv.append(f"--dependency=afternotok:{bundle.jobid}")
    argv.append(str(path))

    sub = run(argv, timeout=cfg.cmd_timeout, cwd=workdir)
    if not sub.ok:
        return HealOutcome(False, f"sbatch failed: {(sub.stderr or sub.stdout).strip()}",
                           patch_file=str(path), fix_kind=fix.kind)
    m = re.search(r"Submitted batch job (\d+)", sub.stdout)
    new_id = m.group(1) if m else "?"
    _record_chain(cfg, new_id, bundle.jobid, fix.kind, str(path))
    return HealOutcome(True, f"resubmitted as job {new_id} with {fix.kind}",
                       new_jobid=new_id, patch_file=str(path), fix_kind=fix.kind)
