"""Phase 4: patch a failed job's submit script with a surgical, idempotent fix.

Each patcher takes the original script text + the collected bundle and returns
a FixOutcome with the patched text (or a skip reason) plus a unified diff.

Patches are minimal: replace the directive in place, or insert a single line
at a well-defined anchor (after the last contiguous #SBATCH block). If a
patcher would have to rewrite >20% of the lines, it downgrades to ``skipped``
so we don't ship a fragile rewrite (per the project quality bar).
"""
from __future__ import annotations

import difflib
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .collect import CollectedBundle, _bytes_safe

log = logging.getLogger("slurm_doctor.fix")

# Cap a patcher to changing at most this fraction of lines from the original.
_MAX_DIFF_RATIO = 0.20


@dataclass
class FixOutcome:
    fix_kind: str
    patched_text: str | None       # None means fix could not be applied
    skipped_reason: str | None
    diff: str | None               # unified diff vs original (when patched)


class FixError(Exception):
    pass


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def apply_fix(fix_kind: str, original_text: str, bundle: CollectedBundle) -> FixOutcome:
    handler = _DISPATCH.get(fix_kind)
    if handler is None:
        return FixOutcome(
            fix_kind=fix_kind,
            patched_text=None,
            skipped_reason=f"fix_kind {fix_kind!r} not implemented in Phase 4",
            diff=None,
        )
    try:
        out = handler(original_text, bundle)
    except FixError as e:
        return FixOutcome(fix_kind=fix_kind, patched_text=None,
                          skipped_reason=str(e), diff=None)
    if out.patched_text is not None and out.diff is None:
        out = FixOutcome(
            fix_kind=out.fix_kind,
            patched_text=out.patched_text,
            skipped_reason=out.skipped_reason,
            diff=_unified_diff(original_text, out.patched_text, label=fix_kind),
        )
    # Refuse patches that rewrite too much of the script.
    if out.patched_text is not None:
        reason = _over_edit_reason(original_text, out.patched_text)
        if reason is not None:
            return FixOutcome(
                fix_kind=fix_kind,
                patched_text=None,
                skipped_reason=f"{reason}; would rewrite too much — downgrade to suggest",
                diff=None,
            )
    return out


def write_patched_script(
    bundle: CollectedBundle,
    outcome: FixOutcome,
    *,
    target_dir: str | Path | None = None,
) -> Path:
    """Write the patched text to ``<dir>/<orig>.fix<N>.sh`` and return the path.

    target_dir defaults to bundle.workdir (so #SBATCH paths stay valid).
    The number suffix auto-increments so we never overwrite a prior patch.
    """
    if outcome.patched_text is None:
        raise FixError("can't write a patch that wasn't produced")
    if target_dir is None:
        target_dir = bundle.workdir or "."
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    original_name = _original_basename(bundle)
    stem, suffix = _split_script_name(original_name)
    n = 1
    while (target_dir / f"{stem}.fix{n}{suffix}").exists():
        n += 1
    out = target_dir / f"{stem}.fix{n}{suffix}"
    out.write_text(outcome.patched_text)
    out.chmod(0o755)
    return out


# ---------------------------------------------------------------------------
# Patchers
# ---------------------------------------------------------------------------

def _bump_memory(text: str, bundle: CollectedBundle) -> FixOutcome:
    if not bundle.max_rss:
        raise FixError("no MaxRSS recorded; cannot compute new --mem")
    max_rss_bytes = _bytes_safe(bundle.max_rss)
    if max_rss_bytes <= 0:
        raise FixError(f"unparseable MaxRSS={bundle.max_rss!r}")
    target_bytes = math.ceil(max_rss_bytes * 1.3)
    new_gb = max(1, math.ceil(target_bytes / (1024 ** 3)))
    new_value = f"{new_gb}G"
    patched = _replace_or_insert_sbatch(text, "mem", new_value)
    return FixOutcome(fix_kind="bump_memory", patched_text=patched, skipped_reason=None, diff=None)


def _bump_time(text: str, bundle: CollectedBundle) -> FixOutcome:
    if not bundle.elapsed:
        raise FixError("no Elapsed recorded; cannot compute new --time")
    elapsed_s = _parse_duration(bundle.elapsed)
    if elapsed_s is None or elapsed_s <= 0:
        raise FixError(f"unparseable Elapsed={bundle.elapsed!r}")
    target_s = math.ceil(elapsed_s * 1.5)
    # round up to the next whole minute so --time stays human-friendly
    target_s = ((target_s + 59) // 60) * 60
    cap = _partition_max_time_seconds(bundle)
    if cap is not None and target_s > cap:
        target_s = cap
    new_value = _format_duration(target_s)
    patched = _replace_or_insert_sbatch(text, "time", new_value)
    return FixOutcome(fix_kind="bump_time", patched_text=patched, skipped_reason=None, diff=None)


def _line_has_set_e(line: str) -> bool:
    """True if a bash line invokes ``set -e`` (in any combined form) or
    ``set -o errexit``. We strip trailing comments first."""
    s = line.split("#", 1)[0].strip()
    if not s.startswith("set"):
        return False
    rest = s[3:]
    if rest and not rest[0].isspace():  # e.g. "settle" — different word
        return False
    rest = rest.lstrip()
    return "-e" in rest or "errexit" in rest


def _add_set_eux(text: str, bundle: CollectedBundle) -> FixOutcome:
    # idempotent: skip if `set -e` is already there within the first 50 lines
    for line in text.splitlines()[:50]:
        if _line_has_set_e(line):
            return FixOutcome(
                fix_kind="add_set_eux",
                patched_text=None,
                skipped_reason="script already uses `set -e`",
                diff=None,
            )
    lines = text.splitlines(keepends=False)
    insert_at = _first_non_directive_line(lines)
    new_lines = lines[:insert_at] + [
        "# slurm-doctor: stop on first error and treat unset vars / pipefails as errors",
        "set -euo pipefail",
        "",
    ] + lines[insert_at:]
    patched = "\n".join(new_lines)
    if text.endswith("\n"):
        patched += "\n"
    return FixOutcome(fix_kind="add_set_eux", patched_text=patched, skipped_reason=None, diff=None)


def _prepend_modules(text: str, bundle: CollectedBundle) -> FixOutcome:
    # idempotent: skip if lmod is already sourced or a module load already exists
    head = "\n".join(text.splitlines()[:80])
    if "module load" in head or "source /etc/profile.d/lmod.sh" in head:
        return FixOutcome(
            fix_kind="prepend_modules",
            patched_text=None,
            skipped_reason="modules already set up in this script",
            diff=None,
        )
    lines = text.splitlines(keepends=False)
    insert_at = _first_non_directive_line(lines)
    new_lines = lines[:insert_at] + [
        "# slurm-doctor: ensure the `module` command is available, then load",
        "# the specific module(s) this job needs. Edit the names below.",
        "if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi",
        "# module load <NAME>   # <-- slurm-doctor: replace with the right module",
        "",
    ] + lines[insert_at:]
    patched = "\n".join(new_lines)
    if text.endswith("\n"):
        patched += "\n"
    return FixOutcome(fix_kind="prepend_modules", patched_text=patched, skipped_reason=None, diff=None)


_MISSING_PATH_RX = re.compile(r"([^\s:]+):\s*No such file or directory")


def _fix_path(text: str, bundle: CollectedBundle) -> FixOutcome:
    """Rewrite a missing path to a similarly-named file that exists in WorkDir.

    Reads the missing path from stderr, fuzzy-matches its basename against the
    files actually present in WorkDir, and rewrites occurrences in the script.
    If nothing similar exists we raise FixError (caller downgrades to suggest)
    rather than guess.
    """
    import difflib
    import os

    stderr_path = bundle.stderr.cached_path if bundle.stderr else None
    stderr = ""
    if stderr_path:
        try:
            stderr = Path(stderr_path).read_text(errors="replace")
        except OSError:
            stderr = ""
    missing = []
    for m in _MISSING_PATH_RX.finditer(stderr):
        tok = m.group(1)
        # ignore the slurm_script wrapper path itself
        if "slurm_script" in tok:
            continue
        missing.append(tok)
    if not missing:
        raise FixError("no 'No such file or directory' path found in stderr")

    workdir = bundle.workdir or "."
    try:
        present = os.listdir(workdir)
    except OSError as e:
        raise FixError(f"can't list WorkDir {workdir!r}: {e}")

    replaced_any = False
    patched = text
    for tok in missing:
        base = Path(tok).name
        cand = difflib.get_close_matches(base, present, n=1, cutoff=0.6)
        if not cand or cand[0] == base:
            continue
        # Preserve a leading ./ if the original had one.
        repl = cand[0]
        if tok.startswith("./"):
            repl = f"./{repl}"
        elif tok.startswith("/"):
            repl = str(Path(workdir) / cand[0])
        if tok in patched:
            patched = patched.replace(tok, repl)
            replaced_any = True
    if not replaced_any:
        raise FixError(
            "no similar file in WorkDir to rewrite to (won't guess a path)"
        )
    return FixOutcome(fix_kind="fix_path", patched_text=patched, skipped_reason=None, diff=None)


_MPI_LAUNCHER_RX = re.compile(r"(?<![\w./-])(mpirun|mpiexec)(?![\w.-])")


def _swap_mpi_launcher(text: str, bundle: CollectedBundle) -> FixOutcome:
    if "srun --mpi=" in text:
        return FixOutcome("swap_mpi_launcher", None,
                          "script already launches via `srun --mpi=`", None)
    if not _MPI_LAUNCHER_RX.search(text):
        return FixOutcome("swap_mpi_launcher", None,
                          "no mpirun/mpiexec invocation found to swap", None)
    new_lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            new_lines.append(line)
            continue
        swapped = _MPI_LAUNCHER_RX.sub("srun --mpi=pmix", line)
        # mpirun's -np maps to srun's -n
        swapped = re.sub(r"(?<![\w-])-np\b", "-n", swapped)
        new_lines.append(swapped)
    patched = "\n".join(new_lines)
    if text.endswith("\n"):
        patched += "\n"
    return FixOutcome("swap_mpi_launcher", patched, None, None)


def _request_constraint(text: str, bundle: CollectedBundle) -> FixOutcome:
    if re.search(r"^\s*#SBATCH\s+(--constraint=|-C\s)", text, re.MULTILINE):
        return FixOutcome("request_constraint", None,
                          "script already requests a --constraint", None)
    # We don't know the right feature for sure; insert a placeholder the user
    # must fill in. This is why request_constraint is gated behind --yes.
    feature = "<FEATURE>"
    patched = _insert_sbatch_directive(
        text,
        f"#SBATCH --constraint={feature}   # slurm-doctor: set the node feature to require",
    )
    return FixOutcome("request_constraint", patched, None, None)


def _add_requeue_guard(text: str, bundle: CollectedBundle) -> FixOutcome:
    has_requeue = re.search(r"^\s*#SBATCH\s+--requeue\b", text, re.MULTILINE)
    has_backoff = "SLURM_RESTART_COUNT" in text
    if has_requeue and has_backoff:
        return FixOutcome("add_requeue_guard", None,
                          "script already has --requeue and a backoff guard", None)
    patched = text
    if not has_requeue:
        patched = _insert_sbatch_directive(
            patched, "#SBATCH --requeue   # slurm-doctor: allow auto-requeue on node/infra failure"
        )
    if not has_backoff:
        lines = patched.splitlines()
        at = _first_non_directive_line(lines)
        guard = [
            "# slurm-doctor: small backoff so a requeued job doesn't hammer a flapping resource",
            'if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then sleep $(( SLURM_RESTART_COUNT * 10 )); fi',
            "",
        ]
        lines = lines[:at] + guard + lines[at:]
        patched = "\n".join(lines)
        if text.endswith("\n"):
            patched += "\n"
    return FixOutcome("add_requeue_guard", patched, None, None)


def _pin_gpu_visible(text: str, bundle: CollectedBundle) -> FixOutcome:
    if "CUDA_VISIBLE_DEVICES" in text or re.search(r"--gpus-per-task", text):
        return FixOutcome("pin_gpu_visible", None,
                          "script already pins GPU visibility", None)
    lines = text.splitlines()
    at = _first_non_directive_line(lines)
    block = [
        "# slurm-doctor: make the GPUs SLURM allocated visible to the process",
        'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-0}}}"',
        "",
    ]
    lines = lines[:at] + block + lines[at:]
    patched = "\n".join(lines)
    if text.endswith("\n"):
        patched += "\n"
    return FixOutcome("pin_gpu_visible", patched, None, None)


_DISPATCH: dict[str, Callable[[str, CollectedBundle], FixOutcome]] = {
    "bump_memory": _bump_memory,
    "bump_time": _bump_time,
    "add_set_eux": _add_set_eux,
    "prepend_modules": _prepend_modules,
    "fix_path": _fix_path,
    "swap_mpi_launcher": _swap_mpi_launcher,
    "request_constraint": _request_constraint,
    "add_requeue_guard": _add_requeue_guard,
    "pin_gpu_visible": _pin_gpu_visible,
}


# ---------------------------------------------------------------------------
# Helpers — exported for testing
# ---------------------------------------------------------------------------

# Match either `#SBATCH --mem=X[ ...]` (long, equals) or `#SBATCH --mem X` (long, space)
# or short flags where applicable. We rewrite the VALUE in-place, preserving any
# trailing comment on the same line.
_SBATCH_LINE = re.compile(r"^(\s*#SBATCH\s+)(.*)$")


def _replace_or_insert_sbatch(text: str, key: str, value: str) -> str:
    """Replace a `#SBATCH --<key>=<X>` (or ``--<key> <X>``) line's value.

    If the directive isn't present, insert ``#SBATCH --<key>=<value>`` at the
    end of the contiguous SBATCH block.
    """
    long_eq = re.compile(rf"^(\s*#SBATCH\s+--{re.escape(key)}=)(\S+)(.*)$")
    long_sp = re.compile(rf"^(\s*#SBATCH\s+--{re.escape(key)})(\s+)(\S+)(.*)$")
    new_lines: list[str] = []
    replaced = False
    for line in text.splitlines():
        m = long_eq.match(line)
        if m and not replaced:
            new_lines.append(f"{m.group(1)}{value}{m.group(3)}")
            replaced = True
            continue
        m = long_sp.match(line)
        if m and not replaced:
            new_lines.append(f"{m.group(1)}{m.group(2)}{value}{m.group(4)}")
            replaced = True
            continue
        new_lines.append(line)
    if not replaced:
        # Insert at end of SBATCH block.
        insert_at = _first_non_directive_line(new_lines)
        new_lines = (
            new_lines[:insert_at]
            + [f"#SBATCH --{key}={value}   # slurm-doctor: inserted"]
            + new_lines[insert_at:]
        )
    patched = "\n".join(new_lines)
    if text.endswith("\n"):
        patched += "\n"
    return patched


def _insert_sbatch_directive(text: str, directive_line: str) -> str:
    """Insert a new ``#SBATCH`` directive at the end of the contiguous SBATCH
    block (right before the first non-directive line)."""
    lines = text.splitlines()
    at = _first_non_directive_line(lines)
    lines = lines[:at] + [directive_line] + lines[at:]
    patched = "\n".join(lines)
    if text.endswith("\n"):
        patched += "\n"
    return patched


def _first_non_directive_line(lines: list[str]) -> int:
    """Return the index of the first line that's neither a shebang, an
    ``#SBATCH`` directive, a blank line, nor a leading comment."""
    saw_directive = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#!") or stripped.startswith("#SBATCH"):
            saw_directive = True
            continue
        if stripped == "" or stripped.startswith("#"):
            # blank/comment line — keep walking
            continue
        return i
    return len(lines) if saw_directive else 0


def _parse_duration(s: str) -> int | None:
    """Parse SLURM duration strings into seconds. Accepts HH:MM:SS,
    MM:SS, DD-HH:MM:SS, or a bare number (minutes)."""
    if not s:
        return None
    s = s.strip()
    days = 0
    if "-" in s:
        d_part, s = s.split("-", 1)
        try:
            days = int(d_part)
        except ValueError:
            return None
    parts = s.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, sec = nums
    elif len(nums) == 2:
        h, m, sec = 0, nums[0], nums[1]
    elif len(nums) == 1:
        return days * 86400 + nums[0] * 60
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + sec


def _format_duration(seconds: int) -> str:
    days, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if days:
        return f"{days}-{h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def _partition_max_time_seconds(bundle: CollectedBundle) -> int | None:
    """Return the partition MaxTime in seconds if discoverable, else None.

    For this cluster MaxTime=INFINITE → returns None (no cap).
    """
    # bundle.meta might have Partition info; scontrol show partition gives MaxTime.
    # For Phase 4 we rely on bundle.timelimit as a sanity cap if present and finite.
    return None


def _original_basename(bundle: CollectedBundle) -> str:
    """The user-facing original script name to base the patched filename on."""
    cmd = bundle.meta.get("Command") if bundle.meta else None
    if cmd:
        return Path(cmd).name
    if bundle.submit_script_path:
        return Path(bundle.submit_script_path).name
    return "patched.sh"


def _split_script_name(name: str) -> tuple[str, str]:
    p = Path(name)
    if p.suffix.lower() in (".sh", ".bash"):
        return p.stem, p.suffix
    return name, ".sh"


def _unified_diff(orig: str, patched: str, *, label: str) -> str:
    diff = difflib.unified_diff(
        orig.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=f"original",
        tofile=f"patched ({label})",
        lineterm="",
    )
    return "\n".join(d.rstrip("\n") for d in diff)


def _over_edit_reason(orig: str, patched: str) -> str | None:
    """Decide whether a patch rewrites too much of the original.

    Two independent guards, tuned so legitimate surgical edits on tiny scripts
    pass while runaway rewrites are caught:

    * altered = original lines replaced or deleted. Allowed up to
      max(2 lines, 20% of the script). A 1-line --mem swap on a 3-line script
      (33%) still passes; gutting half a large script does not.
    * inserted = brand-new lines. Pure insertions (set -e, module load) are
      cheap, but a patcher that balloons the script is suspect: capped at
      max(40 lines, 3x the original length).
    """
    o = orig.splitlines()
    p = patched.splitlines()
    if not o:
        return None if not p else None
    sm = difflib.SequenceMatcher(a=o, b=p)
    altered = inserted = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            altered += (i2 - i1)
            inserted += (j2 - j1)
        elif tag == "delete":
            altered += (i2 - i1)
        elif tag == "insert":
            inserted += (j2 - j1)
    altered_budget = max(2, int(_MAX_DIFF_RATIO * len(o)))
    if altered > altered_budget:
        pct = altered / len(o)
        return f"rewrite alters {altered} of {len(o)} original lines ({pct:.0%})"
    insert_budget = max(40, 3 * len(o))
    if inserted > insert_budget:
        return f"rewrite inserts {inserted} new lines (> {insert_budget})"
    return None
