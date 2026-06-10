"""Shared utilities: the single shell-out wrapper, size caps, redaction,
and SLURM unit parsing (memory / elapsed / timestamps).

Every external command in slurm-doctor goes through :func:`run`.  It logs the
command, captures stdout/stderr, enforces a timeout and never uses
``shell=True``.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("slurm_doctor")

DEFAULT_TIMEOUT = int(os.environ.get("SLURM_DOCTOR_CMD_TIMEOUT", "30"))


@dataclasses.dataclass
class CmdResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class CommandError(RuntimeError):
    def __init__(self, result: CmdResult):
        self.result = result
        super().__init__(
            f"command failed ({result.returncode}"
            f"{', timed out' if result.timed_out else ''}): "
            f"{' '.join(result.argv)}\nstderr: {result.stderr[-500:]}"
        )


def run(
    argv: list[str],
    *,
    timeout: float | None = None,
    check: bool = False,
    cwd: str | os.PathLike | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> CmdResult:
    """Run *argv* (a list, never a shell string) with logging and a timeout."""
    if isinstance(argv, str):  # defence in depth: refuse shell strings
        raise TypeError("run() takes a list of arguments, not a string")
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            input=input_text,
            shell=False,
        )
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = 124
        out = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    except FileNotFoundError:
        rc, out, err = 127, "", f"{argv[0]}: command not found"
    dur = time.monotonic() - t0
    result = CmdResult(list(argv), rc, out, err, dur, timed_out)
    log.debug("run %s -> rc=%s (%.2fs)", " ".join(argv), rc, dur)
    if check and not result.ok:
        raise CommandError(result)
    return result


# --------------------------------------------------------------------------
# Size-capped reads (head + tail strategy)
# --------------------------------------------------------------------------

def head_tail_cap(text: str, max_bytes: int) -> tuple[str, bool]:
    """Cap *text* to roughly *max_bytes*, keeping the head and the tail.

    Returns ``(capped_text, was_truncated)``.  The tail gets 2/3 of the
    budget because failures usually live at the end of a log.
    """
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text, False
    head_n = max_bytes // 3
    tail_n = max_bytes - head_n
    head = raw[:head_n].decode("utf-8", errors="replace")
    tail = raw[-tail_n:].decode("utf-8", errors="replace")
    # cut at line boundaries so quoted evidence lines stay intact
    head = head.rsplit("\n", 1)[0]
    tail = tail.split("\n", 1)[-1]
    marker = (
        f"\n... [slurm-doctor: truncated {len(raw) - max_bytes} bytes] ...\n"
    )
    return head + marker + tail, True


def read_file_capped(path: str | Path, max_bytes: int) -> tuple[str | None, dict]:
    """Read *path*, applying the head+tail cap. Returns (text, meta)."""
    p = Path(path)
    meta: dict = {"path": str(p)}
    try:
        size = p.stat().st_size
    except OSError as exc:
        meta["error"] = str(exc)
        return None, meta
    meta["size_bytes"] = size
    try:
        if size <= max_bytes:
            return p.read_text(errors="replace"), meta
        with p.open("rb") as fh:
            head = fh.read(max_bytes // 3)
            fh.seek(-(max_bytes - max_bytes // 3), os.SEEK_END)
            tail = fh.read()
        text = (
            head.decode("utf-8", errors="replace").rsplit("\n", 1)[0]
            + f"\n... [slurm-doctor: truncated {size - max_bytes} bytes] ...\n"
            + tail.decode("utf-8", errors="replace").split("\n", 1)[-1]
        )
        meta["truncated"] = True
        return text, meta
    except OSError as exc:
        meta["error"] = str(exc)
        return None, meta


# --------------------------------------------------------------------------
# Secret redaction (applied to anything that leaves the box, e.g. LLM calls)
# --------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"(?im)^(?P<k>\s*(?:export\s+)?[A-Za-z0-9_]*(?:AWS_[A-Z0-9_]+|[A-Za-z0-9_]*TOKEN[A-Za-z0-9_]*|[A-Za-z0-9_]*_KEY|[A-Za-z0-9_]*PASSWORD[A-Za-z0-9_]*|[A-Za-z0-9_]*SECRET[A-Za-z0-9_]*|[A-Za-z0-9_]*CREDENTIALS?[A-Za-z0-9_]*))(?P<sep>\s*=\s*)(?P<v>.+)$"),
    re.compile(r"(?i)\b(authorization|x-api-key|api[-_]?key|bearer)\b(\s*[:=]\s*)(\S+)"),
]


def redact(text: str) -> str:
    """Blank out values of likely-secret variables and auth headers."""
    out = _SECRET_PATTERNS[0].sub(lambda m: m.group("k") + m.group("sep") + "[REDACTED]", text)
    out = _SECRET_PATTERNS[1].sub(lambda m: m.group(1) + m.group(2) + "[REDACTED]", out)
    return out


# --------------------------------------------------------------------------
# SLURM unit parsing
# --------------------------------------------------------------------------

_MEM_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGTP]?)([BncCN]?)\s*$", re.I)


def parse_mem_to_bytes(value: str | None) -> int | None:
    """Parse SLURM memory strings: ``4000M``, ``2G``, ``102400K``, ``312456Kn``.

    sacct ReqMem may carry a ``n`` (per node) / ``c`` (per CPU) suffix; the
    multiplier semantics are left to the caller — this returns plain bytes.
    """
    if not value:
        return None
    m = _MEM_RE.match(value)
    if not m:
        return None
    num = float(m.group(1))
    unit = (m.group(2) or "M").upper()  # bare numbers are MB in sbatch
    mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}[unit or "M"]
    return int(num * mult)


def mem_suffix(value: str | None) -> str:
    """Return the per-node/per-cpu suffix of a ReqMem value ('n', 'c' or '')."""
    if not value:
        return ""
    m = _MEM_RE.match(value)
    if not m:
        return ""
    suf = m.group(3) or ""
    return suf.lower() if suf.lower() in ("n", "c") else ""


def format_mem_mb(n_bytes: int) -> str:
    """Format bytes as an sbatch --mem value, rounded UP to the next GB
    (or to the next 64M below 1G so tiny jobs don't balloon)."""
    gb = 1024**3
    if n_bytes >= gb:
        return f"{math.ceil(n_bytes / gb)}G"
    mb = math.ceil(n_bytes / (64 * 1024**2)) * 64
    return f"{mb}M"


_ELAPSED_RE = re.compile(
    r"^\s*(?:(?P<days>\d+)-)?(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+)(?:\.\d+)?\s*$"
)


def parse_elapsed_to_seconds(value: str | None) -> int | None:
    """Parse ``[DD-[HH:]]MM:SS`` (sacct Elapsed/Timelimit). Returns seconds."""
    if not value or value in ("UNLIMITED", "Partition_Limit", "INVALID", "Unknown"):
        return None
    m = _ELAPSED_RE.match(value)
    if not m:
        return None
    days = int(m.group("days") or 0)
    h = int(m.group("h") or 0)
    return ((days * 24 + h) * 60 + int(m.group("m"))) * 60 + int(m.group("s"))


def format_timelimit(seconds: int) -> str:
    """Format seconds as sbatch --time ``D-HH:MM:SS`` (or ``HH:MM:SS``)."""
    seconds = max(60, int(seconds))
    days, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if days:
        return f"{days}-{h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_slurm_timestamp(value: str | None) -> datetime | None:
    """Parse sacct/scontrol timestamps like ``2026-06-10T15:04:05``."""
    if not value or value in ("Unknown", "None", "N/A"):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


_SINCE_RE = re.compile(
    r"^\s*(\d+)\s*(second|sec|s|minute|min|m|hour|hr|h|day|d|week|w)s?\s*ago\s*$", re.I
)


def parse_since(value: str, now: datetime | None = None) -> datetime:
    """Parse '--since' values: '1 hour ago', '30 min ago', ISO timestamps,
    or sacct-style 'now-2hours'."""
    now = now or datetime.now()
    value = value.strip()
    m = _SINCE_RE.match(value)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()[0]
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
        if m.group(2).lower() in ("min", "minute"):
            mult = 60
        return now - timedelta(seconds=n * mult)
    m = re.match(r"^now-(\d+)(second|minute|hour|day|week)s?$", value, re.I)
    if m:
        mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800}[
            m.group(2).lower()
        ]
        return now - timedelta(seconds=int(m.group(1)) * mult)
    ts = parse_slurm_timestamp(value)
    if ts:
        return ts
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"cannot parse --since value: {value!r}")


def expand_filename_pattern(pattern: str, meta: dict) -> str:
    """Expand the common sbatch filename patterns (%j, %x, %u, %N, %A, %a)."""
    def sub(m: re.Match) -> str:
        c = m.group(1)
        return {
            "%": "%",
            "j": str(meta.get("jobid", "")),
            "J": str(meta.get("jobid", "")),
            "x": str(meta.get("jobname", "")),
            "u": str(meta.get("user", "")),
            "N": str(meta.get("node", "")),
            "A": str(meta.get("array_job_id", meta.get("jobid", ""))),
            "a": str(meta.get("array_task_id", "")),
        }.get(c, m.group(0))

    return re.sub(r"%(.)", sub, pattern)
