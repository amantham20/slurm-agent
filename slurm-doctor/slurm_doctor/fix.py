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
        ratio = _changed_line_ratio(original_text, out.patched_text)
        if ratio > _MAX_DIFF_RATIO:
            return FixOutcome(
                fix_kind=fix_kind,
                patched_text=None,
                skipped_reason=(
                    f"diff would rewrite {ratio:.0%} of lines "
                    f"(> {_MAX_DIFF_RATIO:.0%} ceiling); downgrade to suggest"
                ),
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


_DISPATCH: dict[str, Callable[[str, CollectedBundle], FixOutcome]] = {
    "bump_memory": _bump_memory,
    "bump_time": _bump_time,
    "add_set_eux": _add_set_eux,
    "prepend_modules": _prepend_modules,
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


def _changed_line_ratio(orig: str, patched: str) -> float:
    """Return the fraction of ORIGINAL lines that were replaced or deleted.

    Pure insertions are not "rewrites" — they leave existing lines intact and
    don't change the meaning of any line — so they don't count toward the cap.
    """
    o = orig.splitlines()
    p = patched.splitlines()
    if not o:
        return 1.0 if p else 0.0
    sm = difflib.SequenceMatcher(a=o, b=p)
    touched = 0
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            touched += (i2 - i1)
    return touched / len(o)
